"""Approval recovery through PostgreSQL checkpoints across RunManager instances."""

import asyncio
import json
import os
import time
from types import SimpleNamespace

import psycopg
import pytest

from aegisflow.api.runs import RunManager
from aegisflow.orchestrator.ledger import postgres_checkpointer
from aegisflow.rag.qa_workflow import build_qa_workflow


DSN = os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")


def postgres_available() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not postgres_available(), reason="Postgres not reachable")


class FakeApprovalClient:
    async def complete(self, messages, **_options):
        prompt = "\n".join(message.content for message in messages)
        if "Classify the user's question" in prompt:
            body = {"route": "multi_hop"}
        elif "Break the question into 2 to 4" in prompt:
            body = {"sub_questions": ["part one", "part two", "part three", "part four"]}
        elif "untrusted_evidence" in prompt:
            return SimpleNamespace(
                text="A grounded answer [1].",
                usage=SimpleNamespace(total_tokens=1, prompt_tokens=1, completion_tokens=0),
                attempts=1,
                finish_reason="stop",
            )
        else:
            raise AssertionError("unexpected fake-client request")
        return SimpleNamespace(
            text=json.dumps(body),
            usage=SimpleNamespace(total_tokens=1, prompt_tokens=1, completion_tokens=0),
            attempts=1,
            finish_reason="stop",
        )


async def evidence_retriever(_question):
    return [{"chunk_id": "fixture/doc#0", "source": "fixture", "heading_path": [], "text": "grounding fact"}]


async def always_faithful(*_args):
    return SimpleNamespace(
        score=1.0,
        verdict=SimpleNamespace(unsupported=[]),
        tokens_used=1,
        attempts=1,
    )


async def wait_for_status(manager: RunManager, run_id: str, wanted: str, timeout: float = 10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = await manager.status(run_id)
        if result and result["status"] == wanted:
            return result
        await asyncio.sleep(0.025)
    raise AssertionError(f"run did not reach {wanted}")


@pytest.mark.asyncio
async def test_approval_survives_postgres_checkpoint_and_new_run_manager():
    fake_client = FakeApprovalClient()

    def make_factory(saver):
        def qa_factory():
            return build_qa_workflow(
                fake_client,
                evidence_retriever,
                checkpointer=saver,
                faithfulness_verifier=always_faithful,
            )
        return qa_factory

    async with postgres_checkpointer(DSN) as first_saver:
        qa_factory = make_factory(first_saver)
        first_manager = RunManager(first_saver, qa_factory=qa_factory)
        run_id = first_manager.start("A multi-part question", graph="qa")
        paused = await wait_for_status(first_manager, run_id, "awaiting_approval")
        assert paused["approval"]["kind"] == "qa_plan_approval"
        await first_manager.aclose()

    # Reopen the AsyncPostgresSaver as well as constructing a new manager. The
    # interrupt must be restored from persisted PostgreSQL state only.
    async with postgres_checkpointer(DSN) as second_saver:
        second_manager = RunManager(second_saver, qa_factory=make_factory(second_saver))
        restored = await second_manager.status(run_id)
        assert restored["status"] == "awaiting_approval"
        assert restored["sub_questions"] == ["part one", "part two", "part three", "part four"]
        await second_manager.approve(run_id, True)
        record = second_manager.records[run_id]
        await asyncio.wait_for(record.task, timeout=10)
        completed = await second_manager.status(run_id)
        assert completed["status"] == "completed"
        assert completed["final_answer"] == "A grounded answer [1]."
        await second_manager.aclose()
