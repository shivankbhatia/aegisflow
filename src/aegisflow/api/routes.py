import asyncio
import json
import logging
import os
import re
import secrets
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from ..llm_gateway import Message, from_env
from ..rag.runtime import ingest_local_corpus

log = logging.getLogger("aegisflow.api.rag")

router = APIRouter()


class WorkflowRequest(BaseModel):
    task_description: str = Field(min_length=1)


class RunResponse(BaseModel):
    run_id: str
    status: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=10)


_CITATION = re.compile(r"\[(\d+)\]")
_SYSTEM_PROMPT = (
    "Answer the user's question using only the numbered contexts below. Cite each factual claim "
    "with its context number like [1]. If the contexts do not contain the answer, say exactly "
    "'insufficient evidence'. Do not invent facts or citations."
)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/query")
async def query_rag(body: QueryRequest, request: Request):
    runtime = getattr(request.app.state, "rag", None)
    if runtime is None:
        raise HTTPException(503, "RAG is disabled; set AEGIS_RAG_ENABLED=1 and restart")
    gateway = getattr(request.app.state, "gateway", None)
    if gateway is None:
        gateway = from_env()
        request.app.state.gateway = gateway
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    retrieve_start = start
    try:
        timing = {"retrieval_ms": 0.0, "rerank_ms": 0.0}
        def record_timings(retrieval_ms, rerank_ms):
            timing["retrieval_ms"], timing["rerank_ms"] = retrieval_ms, rerank_ms
        try:
            ids = await runtime.retrieve(
                body.query,
                body.k,
                timing_callback=record_timings,
                candidate_limit=body.k,
            )
        except TypeError as exc:
            if "timing_callback" not in str(exc):
                raise
            ids = await runtime.retrieve(body.query, body.k, candidate_limit=body.k)
        total_retrieval_pipeline_ms = (time.perf_counter() - retrieve_start) * 1000
        retrieval_ms = timing["retrieval_ms"] or total_retrieval_pipeline_ms
        rerank_ms = timing["rerank_ms"]
        chunks = await asyncio.to_thread(
            runtime.index.fetch_chunks, ids,
            strategy=getattr(runtime, "strategy", "structure-aware"),
        )
        contexts = [(chunk_id, chunks[chunk_id]) for chunk_id in ids if chunk_id in chunks]
    except Exception as exc:
        log.exception("rag_request_failed request_id=%s stage=retrieval", request_id)
        raise HTTPException(503, "Retrieval failed") from exc
    prompt_context = "\n\n".join(
        f"[{i}] source={item['source']} heading={' > '.join(item['heading_path'])}\n{item['text']}"
        for i, (_chunk_id, item) in enumerate(contexts, 1)
    )
    messages = [Message("system", _SYSTEM_PROMPT), Message("user", f"Question: {body.query}\n\nContexts:\n{prompt_context}")]

    async def events():
        answer_parts: list[str] = []
        first_token_ms = None
        error = None
        try:
            async for token in gateway.stream(
                messages,
                max_tokens=int(os.getenv("AEGIS_RAG_MAX_TOKENS", "1200")),
                reasoning_effort=os.getenv("AEGIS_RAG_REASONING_EFFORT", "low"),
            ):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - start) * 1000
                answer_parts.append(token)
                yield _sse("token", {"request_id": request_id, "text": token})
            answer = "".join(answer_parts)
            cited = sorted({int(num) for num in _CITATION.findall(answer)})
            valid = [num for num in cited if 1 <= num <= len(contexts)]
            invalid = [num for num in cited if num < 1 or num > len(contexts)]
            citations = [
                {"number": num, "chunk_id": contexts[num - 1][0],
                 "title": contexts[num - 1][1]["doc_id"].rsplit("/", 1)[-1],
                 "source": contexts[num - 1][1]["source"],
                 "heading_path": contexts[num - 1][1]["heading_path"]}
                for num in valid
            ]
            yield _sse("citations", {"request_id": request_id, "citations": citations, "invalid_markers": invalid})
            token_count = gateway.count_tokens(messages) + gateway.count_tokens(answer)
            cost_proxy = token_count * float(os.getenv("AEGIS_COST_PER_TOKEN_USD", "0.0000001"))
            total_ms = (time.perf_counter() - start) * 1000
            log.info(
                "rag_request request_id=%s retrieval_ms=%.1f rerank_ms=%.1f first_token_ms=%s tokens=%d cost_proxy_usd=%.8f total_ms=%.1f",
                request_id, retrieval_ms, rerank_ms,
                f"{first_token_ms:.1f}" if first_token_ms is not None else "none",
                token_count, cost_proxy, total_ms,
            )
            yield _sse("done", {
                "request_id": request_id,
                "tokens": token_count,
                "cost_proxy_usd": cost_proxy,
                "timings_ms": {
                    "retrieval": retrieval_ms,
                    "rerank": rerank_ms,
                    "first_token": first_token_ms,
                    "total": total_ms,
                },
            })
        except Exception as exc:
            error = str(exc)
            log.exception("rag_request_failed request_id=%s stage=generation", request_id)
            elapsed_ms = (time.perf_counter() - start) * 1000
            yield _sse("error", {
                "request_id": request_id,
                "detail": error,
                "timings_ms": {
                    "retrieval": retrieval_ms,
                    "rerank": rerank_ms,
                    "first_token": first_token_ms,
                    "elapsed": elapsed_ms,
                },
            })

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Request-ID": request_id})


async def _run_ingest(request: Request, job_id: str) -> None:
    jobs = request.app.state.ingest_jobs
    jobs[job_id] = {"status": "running"}
    try:
        result = await asyncio.to_thread(ingest_local_corpus, request.app.state.rag)
        jobs[job_id] = {"status": "completed", **result}
    except Exception as exc:
        log.exception("rag_ingest_failed job_id=%s", job_id)
        jobs[job_id] = {"status": "failed", "error": str(exc)}


def _require_ingest_token(request: Request, token: str | None) -> None:
    expected = os.getenv("AEGIS_INGEST_TOKEN")
    if not expected:
        raise HTTPException(503, "Ingestion is disabled until AEGIS_INGEST_TOKEN is configured")
    if token is None or not secrets.compare_digest(token, expected):
        raise HTTPException(401, "Invalid or missing X-Ingest-Token")


@router.post("/ingest", status_code=202)
async def ingest_rag(
    background_tasks: BackgroundTasks,
    request: Request,
    x_ingest_token: str | None = Header(default=None, alias="X-Ingest-Token"),
):
    _require_ingest_token(request, x_ingest_token)
    if getattr(request.app.state, "rag", None) is None:
        raise HTTPException(503, "RAG is disabled; set AEGIS_RAG_ENABLED=1 and restart")
    if not hasattr(request.app.state, "ingest_jobs"):
        request.app.state.ingest_jobs = {}
    job_id = str(uuid.uuid4())
    request.app.state.ingest_jobs[job_id] = {"status": "queued"}
    background_tasks.add_task(_run_ingest, request, job_id)
    return {"job_id": job_id, "status": "queued"}


@router.get("/ingest/{job_id}")
async def get_ingest_job(
    job_id: str,
    request: Request,
    x_ingest_token: str | None = Header(default=None, alias="X-Ingest-Token"),
):
    _require_ingest_token(request, x_ingest_token)
    job = getattr(request.app.state, "ingest_jobs", {}).get(job_id)
    if job is None:
        raise HTTPException(404, "unknown ingestion job")
    return {"job_id": job_id, **job}


@router.post("/run", response_model=RunResponse, status_code=202)
async def trigger_workflow(body: WorkflowRequest, request: Request):
    run_id = request.app.state.runs.start(body.task_description)
    return {"run_id": run_id, "status": "queued"}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request):
    info = await request.app.state.runs.status(run_id)
    if info is None:
        raise HTTPException(404, "unknown run")
    return info


@router.post("/runs/{run_id}/resume", response_model=RunResponse, status_code=202)
async def resume_run(run_id: str, request: Request):
    runs = request.app.state.runs
    info = await runs.status(run_id)
    if info is None:
        raise HTTPException(404, "unknown run")
    if info["status"] not in ("interrupted", "failed"):
        raise HTTPException(409, f"run is {info['status']}; only interrupted or failed runs resume")
    runs.resume(run_id)
    return {"run_id": run_id, "status": "queued"}


@router.websocket("/ws/events/{run_id}")
async def events(websocket: WebSocket, run_id: str):
    """Replays this process's events for the run, then streams new ones."""
    await websocket.accept()
    rec = websocket.app.state.runs.records.get(run_id)
    if rec is None:
        await websocket.send_json(
            {"event": "error", "detail": "no live events for this run; use GET /runs/{id}"}
        )
        await websocket.close()
        return
    i = 0
    try:
        while True:
            async with rec.cond:
                await rec.cond.wait_for(lambda: len(rec.events) > i or rec.finished)
            while i < len(rec.events):
                await websocket.send_json(rec.events[i])
                i += 1
            if rec.finished and i >= len(rec.events):
                break
        await websocket.close()
    except WebSocketDisconnect:
        pass
