from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

router = APIRouter()


class WorkflowRequest(BaseModel):
    task_description: str = Field(min_length=1)


class RunResponse(BaseModel):
    run_id: str
    status: str


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