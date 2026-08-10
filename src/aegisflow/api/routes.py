from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()

class WorkflowRequest(BaseModel):
    task_description: str
    repository_url: str | None = None

class WorkflowResponse(BaseModel):
    run_id: str
    status: str

@router.post("/run", response_model=WorkflowResponse)
async def trigger_workflow(request: WorkflowRequest):
    # TODO: Initialize LangGraph workflow and return run_id
    run_id = "test-run-123"
    return {"run_id": run_id, "status": "queued"}

@router.websocket("/ws/events/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await websocket.accept()
    try:
        # TODO: Stream events from the state ledger for the given run_id
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        pass
