from types import SimpleNamespace

import pytest

from aegisflow.orchestrator.guardrails import GuardrailsInterceptor
from aegisflow.orchestrator.workflow import run_workflow


def reject_until(n_ok):
    return lambda n, code_patch, test_patch: SimpleNamespace(approved=n >= n_ok, feedback="fix it")


@pytest.mark.asyncio
async def test_approved_first_try(make_agents_factory):
    agents = make_agents_factory()
    result = await run_workflow("t", agents=agents)
    assert result["approved"] is True and agents.coder.calls == 1
    assert result["tokens_used"] == 30  # 3 calls x 10


@pytest.mark.asyncio
async def test_one_rejection_then_approval(make_agents_factory):
    agents = make_agents_factory(review_fn=reject_until(2))
    result = await run_workflow("t", agents=agents)
    assert result["approved"] is True and agents.coder.calls == 2
    assert result["recursion_count"] == 2


@pytest.mark.asyncio
async def test_circuit_breaker_trips_after_max_attempts(make_agents_factory):
    agents = make_agents_factory(review_fn=reject_until(99))
    with pytest.raises(Exception, match="Maximum recursion limit"):
        await run_workflow("t", agents=agents)
    assert agents.coder.calls == 3


@pytest.mark.asyncio
async def test_coder_receives_feedback_on_retry(make_agents_factory):
    seen = []

    def coder_fn(n, plan, files, **kw):
        seen.append(kw)
        return SimpleNamespace(code_patch=f"c{n}", test_patch=f"t{n}")

    agents = make_agents_factory(coder_fn=coder_fn, review_fn=reject_until(2))
    await run_workflow("t", agents=agents)
    assert seen[0]["feedback"] == "" and seen[1]["feedback"] == "fix it"
    assert seen[1]["previous_code"] == "c1"


@pytest.mark.asyncio
async def test_token_budget_trips(make_agents_factory):
    agents = make_agents_factory(tokens=100)
    with pytest.raises(Exception, match="Budget Exceeded"):
        await run_workflow("t", agents=agents, guardrails=GuardrailsInterceptor(max_tokens=150))
    assert agents.coder.calls == 1  # triage 100 ok, coder pushes total to 200