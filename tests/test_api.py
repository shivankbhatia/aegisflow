"""API integration: real FastAPI app, real Postgres checkpointer, fake agents."""
import os
import time
import json
from types import SimpleNamespace

import psycopg
import pytest
from fastapi.testclient import TestClient

from aegisflow.api.main import create_app

DSN = os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")


def _pg_up() -> bool:
    try:
        psycopg.connect(DSN, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pg_up(), reason="Postgres not reachable")


def wait_for(client, run_id, wanted, timeout=5.0):
    deadline, last = time.time() + timeout, None
    while time.time() < deadline:
        last = client.get(f"/runs/{run_id}").json()
        if last["status"] in wanted:
            return last
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {wanted}; last={last}")


def test_happy_path_and_error_codes(make_agents_factory):
    agents = make_agents_factory()
    with TestClient(create_app(DSN, lambda g: agents)) as client:
        r = client.post("/run", json={"task_description": "fix it"})
        assert r.status_code == 202
        run_id = r.json()["run_id"]

        done = wait_for(client, run_id, {"completed"})
        assert done["approved"] is True and done["plan"] == "p"
        assert done["tokens_used"] == 30 and done["attempts"] == 1

        assert client.get("/runs/does-not-exist").status_code == 404
        assert client.post("/runs/does-not-exist/resume").status_code == 404
        assert client.post(f"/runs/{run_id}/resume").status_code == 409
        assert client.post("/run", json={"task_description": ""}).status_code == 422
        assert client.get("/health").json() == {"status": "healthy"}


def test_failed_run_resumes_without_reexecution(make_agents_factory):
    def crash_once(n, code_patch, test_patch):
        if n == 1:
            raise RuntimeError("worker died")
        return SimpleNamespace(approved=True, feedback="ok")

    agents = make_agents_factory(review_fn=crash_once)
    with TestClient(create_app(DSN, lambda g: agents)) as client:
        run_id = client.post("/run", json={"task_description": "x"}).json()["run_id"]

        failed = wait_for(client, run_id, {"failed"})
        assert "worker died" in failed["error"] and failed["next"] == ["reviewer"]

        assert client.post(f"/runs/{run_id}/resume").status_code == 202
        done = wait_for(client, run_id, {"completed"})
        assert done["approved"] is True and done["error"] is None
        assert (agents.triage.calls, agents.coder.calls, agents.reviewer.calls) == (1, 1, 2)


def test_run_with_missing_credentials_fails_cleanly(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Hermetic: a developer's .env must not re-add the key.
    monkeypatch.setattr("aegisflow.api.main.load_dotenv", lambda: None)
    monkeypatch.setattr("aegisflow.api.runs.load_dotenv", lambda: None)
    with TestClient(create_app(DSN)) as client:  # no agents_factory: real wiring
        run_id = client.post("/run", json={"task_description": "x"}).json()["run_id"]
        failed = wait_for(client, run_id, {"failed"})
        assert "GROQ_API_KEY" in failed["error"]


def test_websocket_streams_node_events_then_done(make_agents_factory):
    agents = make_agents_factory()
    with TestClient(create_app(DSN, lambda g: agents)) as client:
        run_id = client.post("/run", json={"task_description": "x"}).json()["run_id"]
        with client.websocket_connect(f"/ws/events/{run_id}") as ws:
            events = []
            while True:
                ev = ws.receive_json()
                events.append(ev)
                if ev["event"] in ("done", "error"):
                    break
        assert [e["node"] for e in events if e["event"] == "node"] == ["triage", "coder", "reviewer"]
        assert events[-1] == {"seq": 3, "event": "done"}


def test_websocket_unknown_run_reports_error():
    with TestClient(create_app(DSN)) as client:
        with client.websocket_connect("/ws/events/nope") as ws:
            assert ws.receive_json()["event"] == "error"


def test_qa_approval_endpoint_accepts_rejection():
    class FakeQAClient:
        async def complete(self, messages, **_opts):
            prompt = "\n".join(message.content for message in messages)
            if "Decide whether the user's message attempts" in prompt:
                text = json.dumps({"injection": False, "reason": "ordinary question"})
            elif "Classify the user's question" in prompt:
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

    with TestClient(create_app(DSN, qa_client=FakeQAClient(), qa_retriever=fake_retriever)) as client:
        response = client.post("/run", json={"task_description": "Q", "graph": "qa"})
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        paused = wait_for(client, run_id, {"awaiting_approval"})
        assert paused["approval"]["kind"] == "qa_plan_approval"
        rejected = client.post(f"/runs/{run_id}/approve", json={"approve": False})
        assert rejected.status_code == 202
        done = wait_for(client, run_id, {"completed"})
        assert done["status"] == "completed"
        assert done["verdict"] == "refuse"
        assert done["final_answer"] == "insufficient evidence"
