import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..orchestrator.ledger import postgres_checkpointer
from ..llm_gateway import from_env
from ..rag.runtime import load_rag_runtime, rag_config_from_env
from ..rag.qa_workflow import make_retrieval_tool
from .routes import router
from .runs import RunManager


def create_app(
    dsn: str | None = None,
    agents_factory=None,
    *,
    qa_client=None,
    qa_retriever=None,
) -> FastAPI:
    load_dotenv()
    dsn = dsn or os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rag_enabled = os.getenv("AEGIS_RAG_ENABLED", "0").lower() in {"1", "true", "yes"}
        if rag_enabled:
            rag_config_from_env()  # fail before opening resources for invalid settings
        async with postgres_checkpointer(dsn) as saver:
            try:
                effective_qa_client, effective_qa_retriever = qa_client, qa_retriever
                if rag_enabled and (effective_qa_client is None or effective_qa_retriever is None):
                    app.state.rag = await load_rag_runtime(dsn)
                    app.state.gateway = from_env()
                    effective_qa_client = effective_qa_client or app.state.gateway
                    effective_qa_retriever = effective_qa_retriever or make_retrieval_tool(app.state.rag)
                app.state.runs = RunManager(
                    saver, agents_factory,
                    qa_client=effective_qa_client,
                    qa_retriever=effective_qa_retriever,
                )
                app.state.runs.qa_enabled = (
                    effective_qa_client is not None and effective_qa_retriever is not None
                )
                yield
            finally:
                gateway = getattr(app.state, "gateway", None)
                if gateway is not None:
                    await gateway.aclose()
                runs = getattr(app.state, "runs", None)
                if runs is not None:
                    await runs.aclose()

    app = FastAPI(
        title="AegisFlow",
        description="A Deterministic State Harness & Fault-Tolerant Orchestrator for Multi-Agent AI Workflows",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


app = create_app()
