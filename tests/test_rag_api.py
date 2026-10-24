import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegisflow.api.routes import router


class FakeGateway:
    def __init__(self, parts):
        self.parts = parts

    async def stream(self, _messages, **_opts):
        for part in self.parts:
            yield part

    def count_tokens(self, content):
        return len(content) if isinstance(content, str) else sum(len(m.content) for m in content)


class FakeIndex:
    def fetch_chunks(self, ids, *, strategy):
        assert strategy == "structure-aware"
        return {
            chunk_id: {"text": "Context text", "source": "fastapi", "heading_path": ["Background Tasks"], "doc_id": "fastapi/tutorial/background-tasks"}
            for chunk_id in ids
        }


def make_client(parts):
    app = FastAPI()
    app.include_router(router)
    async def retrieve(_query, _k, **kwargs):
        assert kwargs["candidate_limit"] == _k
        kwargs["timing_callback"](12.0, 8.0)
        return ["fastapi/tutorial/background-tasks#0"]
    app.state.rag = SimpleNamespace(retrieve=retrieve, index=FakeIndex())
    app.state.gateway = FakeGateway(parts)
    return TestClient(app)


def parse_sse(response_text):
    out = []
    for frame in response_text.strip().split("\n\n"):
        lines = frame.splitlines()
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


def test_query_sse_order_and_fabricated_citation_validation():
    with make_client(["Answer with valid [1] and fabricated [9]."]) as client:
        response = client.post("/query", json={"query": "background tasks"})
    events = parse_sse(response.text)
    assert response.status_code == 200
    assert [event for event, _ in events] == ["token", "citations", "done"]
    citations = events[1][1]
    assert citations["citations"][0]["chunk_id"] == "fastapi/tutorial/background-tasks#0"
    assert citations["invalid_markers"] == [9]
    assert events[-1][1]["timings_ms"]["retrieval"] == 12.0
    assert events[-1][1]["timings_ms"]["rerank"] == 8.0
    assert events[-1][1]["timings_ms"]["first_token"] is not None


def test_query_sse_preserves_insufficient_evidence_answer():
    with make_client(["insufficient evidence"]) as client:
        response = client.post("/query", json={"query": "unsupported question"})
    events = parse_sse(response.text)
    assert "".join(data["text"] for name, data in events if name == "token") == "insufficient evidence"
    assert [event for event, _ in events] == ["token", "citations", "done"]
