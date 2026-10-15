from types import SimpleNamespace

import pytest

from aegisflow.orchestrator.workflow import Agents


class Fake:
    """Counts calls; optional scripted behaviour per call."""

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def process(self, *a, **kw):
        self.calls += 1
        return self.fn(self.calls, *a, **kw)


def make_agents(review_fn=None, coder_fn=None):
    triage = Fake(lambda n, task: SimpleNamespace(plan="p", required_files=["a.py"]))
    coder = Fake(coder_fn or (lambda n, plan, files: SimpleNamespace(code_patch="c", test_patch="t")))
    reviewer = Fake(review_fn or (lambda n, code_patch, test_patch: SimpleNamespace(approved=True, feedback="ok")))
    return Agents(triage, coder, reviewer)


@pytest.fixture
def make_agents_factory():
    return make_agents