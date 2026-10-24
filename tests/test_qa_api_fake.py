"""QA run API approval flow with an in-memory checkpoint and fake model."""

import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from aegisflow.api.routes import router
from aegisflow.api.runs import RunManager


class FakeQAClient:
    async def complete(self, messages, **_opts):
        prompt = "\n".join(message.content for message in messages)
        if "Classify the user's question" in prompt:
            text = json.dumps({"route": "multi_hop"})
        elif "Break the question into 2 to 4" in prompt:
            text = json.dumps({"sub_questions": ["a", "b", "c", "d"]})
        else:
            raise AssertionError("approval rejection should not synthesize")
        return SimpleNamespace(
            text=text,
            usage=SimpleNamespace(total_tokens=1, prompt_tokens=1, completion_tokens=0),
            attempts=1,
            finish_reason="stop",
        )


async def fake_retriever(_question):
    return []


def test_qa_approval_rest_api_with_fake_dependencies():
    saver = InMemorySaver()
    client = FakeQAClient()
    manager = RunManager(saver, qa_client=client, qa_retriever=fake_retriever)
    app = FastAPI()
    app.state.runs = manager
    app.include_router(router)
    try:
        with TestClient(app) as http:
            response = http.post("/run", json={"task_description": "Q", "graph": "qa"})
            assert response.status_code == 202
            run_id = response.json()["run_id"]
            for _ in range(100):
                paused = http.get(f"/runs/{run_id}").json()
                if paused["status"] == "awaiting_approval":
                    break
                asyncio.run(asyncio.sleep(0.01))
            assert paused["status"] == "awaiting_approval"
            assert paused["approval"]["kind"] == "qa_plan_approval"
            rejected = http.post(f"/runs/{run_id}/approve", json={"approve": False})
            assert rejected.status_code == 202
            for _ in range(100):
                done = http.get(f"/runs/{run_id}").json()
                if done["status"] == "completed":
                    break
                asyncio.run(asyncio.sleep(0.01))
            assert done["status"] == "completed"
            assert done["verdict"] == "refuse"
            assert done["final_answer"] == "insufficient evidence"
    finally:
        asyncio.run(manager.aclose())
