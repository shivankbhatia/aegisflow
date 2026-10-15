from types import SimpleNamespace

import pytest

from aegisflow.orchestrator.workflow import run_workflow


def test_approved_first_try(make_agents_factory):
    agents = make_agents_factory()
    result = run_workflow("t", agents=agents)
    assert result["approved"] is True
    assert agents.coder.calls == 1


def test_one_rejection_then_approval(make_agents_factory):
    agents = make_agents_factory(
        review_fn=lambda n, code_patch, test_patch: SimpleNamespace(approved=n >= 2, feedback="x")
    )
    result = run_workflow("t", agents=agents)
    assert result["approved"] is True
    assert agents.coder.calls == 2
    assert result["recursion_count"] == 2


def test_circuit_breaker_trips_after_max_attempts(make_agents_factory):
    agents = make_agents_factory(
        review_fn=lambda n, code_patch, test_patch: SimpleNamespace(approved=False, feedback="no")
    )
    with pytest.raises(Exception, match="Maximum recursion limit"):
        run_workflow("t", agents=agents)
    assert agents.coder.calls == 3

def test_coder_receives_feedback_on_retry(make_agents_factory):
    seen = []

    def coder_fn(n, plan, files, **kw):
        seen.append(kw)
        return SimpleNamespace(code_patch=f"c{n}", test_patch=f"t{n}")

    agents = make_agents_factory(
        coder_fn=coder_fn,
        review_fn=lambda n, code_patch, test_patch: SimpleNamespace(approved=n >= 2, feedback="fix it"),
    )
    run_workflow("t", agents=agents)
    assert seen[0]["feedback"] == ""
    assert seen[1]["feedback"] == "fix it"
    assert seen[1]["previous_code"] == "c1"