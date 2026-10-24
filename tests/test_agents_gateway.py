"""Real agents + real gateway + real graph; only HTTP and the sandbox are faked."""
import json

import httpx
import pytest

from aegisflow.agents.coder_agent import CoderAgent
from aegisflow.agents.reviewer_agent import ReviewerAgent
from aegisflow.agents.triage_agent import TriageAgent
from aegisflow.llm_gateway import OpenAICompatClient
from aegisflow.orchestrator.guardrails import GuardrailsInterceptor
from aegisflow.orchestrator.workflow import Agents, run_workflow

CODE = "def hello():\n    return 'Hello, World!'"
REPLIES = {
    "Triage Agent": {"rationale": "r", "plan": "edit hello", "required_files": ["hello.py"]},
    "Coder Agent": {"rationale": "r", "code_patch": f"```python\n{CODE}\n```",
                    "test_patch": "assert hello() == 'Hello, World!'"},
    "Reviewer Agent": {"rationale": "r", "approved": True, "feedback": "ok"},
}


def handler(req):
    prompt = json.loads(req.content)["messages"][-1]["content"]
    for key, val in REPLIES.items():
        if key in prompt:
            return httpx.Response(200, json={
                "model": "m",
                "choices": [{"message": {"content": json.dumps(val)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            })
    return httpx.Response(500, text="unrecognised prompt")


class FakeSandbox:
    def __init__(self):
        self.calls = []

    def execute_code(self, code, timeout=10):
        self.calls.append(code)
        return True, ""


def make_client():
    return OpenAICompatClient(api_key="k", base_url="https://x.test/v1", model="m",
                              transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_real_agents_through_gateway_full_graph():
    client, sandbox, g = make_client(), FakeSandbox(), GuardrailsInterceptor()
    agents = Agents(TriageAgent(client), CoderAgent(client),
                    ReviewerAgent(client, sandbox=sandbox, guardrails=g))
    result = await run_workflow("fix hello", agents=agents, guardrails=g)
    assert result["approved"] is True
    assert result["code_patch"] == CODE            # fences stripped
    assert result["tokens_used"] == 90             # 3 calls x (20 + 10)
    assert len(sandbox.calls) == 1 and CODE in sandbox.calls[0]


@pytest.mark.asyncio
async def test_sandbox_results_are_cached_for_identical_code():
    client, sandbox, g = make_client(), FakeSandbox(), GuardrailsInterceptor()
    reviewer = ReviewerAgent(client, sandbox=sandbox, guardrails=g)
    await reviewer.process("x = 1", "assert x == 1")
    await reviewer.process("x = 1", "assert x == 1")
    await reviewer.process("x = 2", "assert x == 2")
    assert len(sandbox.calls) == 2