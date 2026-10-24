"""Kill the reviewer mid-run, resume from Postgres, assert zero re-execution."""
import os
import uuid
from types import SimpleNamespace

import psycopg
import pytest

from aegisflow.orchestrator.ledger import postgres_checkpointer
from aegisflow.orchestrator.workflow import resume_workflow, run_workflow

DSN = os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")


def _pg_up() -> bool:
    try:
        psycopg.connect(DSN, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pg_up(), reason="Postgres not reachable")


class SimulatedCrash(Exception):
    pass


def _crashing_review(n, code_patch, test_patch):
    if n == 1:
        raise SimulatedCrash("process died mid-review")
    return SimpleNamespace(approved=True, feedback="ok")


@pytest.mark.asyncio
async def test_resume_does_not_reexecute_completed_nodes(make_agents_factory):
    thread_id = f"t-{uuid.uuid4()}"
    agents = make_agents_factory(review_fn=_crashing_review)

    async with postgres_checkpointer(DSN) as saver:
        with pytest.raises(SimulatedCrash):
            await run_workflow("task", thread_id=thread_id, agents=agents, checkpointer=saver)

    assert (agents.triage.calls, agents.coder.calls, agents.reviewer.calls) == (1, 1, 1)

    # New connection + new graph object = simulated process restart.
    async with postgres_checkpointer(DSN) as saver:
        result = await resume_workflow(thread_id, checkpointer=saver, agents=agents)

    assert result["approved"] is True
    assert result["tokens_used"] == 30  # triage + coder + resumed reviewer; completed nodes not recounted
    assert agents.triage.calls == 1   # not re-run
    assert agents.coder.calls == 1    # not re-run
    assert agents.reviewer.calls == 2 # failed once, completed on resume