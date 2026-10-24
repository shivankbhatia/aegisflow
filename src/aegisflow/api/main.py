import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..orchestrator.ledger import postgres_checkpointer
from ..rag.runtime import load_rag_runtime
from .routes import router
from .runs import RunManager


def create_app(dsn: str | None = None, agents_factory=None) -> FastAPI:
    load_dotenv()
    dsn = dsn or os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with postgres_checkpointer(dsn) as saver:
            app.state.runs = RunManager(saver, agents_factory)
            try:
                rag_enabled = os.getenv("AEGIS_RAG_ENABLED", "0").lower() in {"1", "true", "yes"}
                if rag_enabled:
                    app.state.rag = await load_rag_runtime(dsn)
                yield
            finally:
                await app.state.runs.aclose()

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
