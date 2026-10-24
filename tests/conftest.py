from types import SimpleNamespace as NS

import pytest

from aegisflow.orchestrator.workflow import Agents


class Fake:
    """Async agent stand-in. fn returns the plain value; usage is attached here."""

    def __init__(self, fn, tokens=10):
        self.fn, self.tokens, self.calls = fn, tokens, 0

    async def process(self, *a, **kw):
        self.calls += 1
        return NS(value=self.fn(self.calls, *a, **kw), usage=NS(total_tokens=self.tokens))


def make_agents(review_fn=None, coder_fn=None, tokens=10):
    triage = Fake(lambda n, task: NS(plan="p", required_files=["a.py"]), tokens)
    coder = Fake(coder_fn or (lambda n, plan, files, **kw: NS(code_patch="c", test_patch="t")), tokens)
    reviewer = Fake(review_fn or (lambda n, code_patch, test_patch: NS(approved=True, feedback="ok")), tokens)
    return Agents(triage, coder, reviewer)


@pytest.fixture
def make_agents_factory():
    return make_agents