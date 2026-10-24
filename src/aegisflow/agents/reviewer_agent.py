import asyncio

from ..llm_gateway import LLMClient, Message, StructuredResult, complete_structured
from ..sandbox.executor import SandboxExecutor
from .base import LLM_OPTS, ReviewerOutputSchema


class ReviewerAgent:

    def __init__(self, client: LLMClient, sandbox=None, guardrails=None):
        self.client = client
        self.sandbox = sandbox or SandboxExecutor()
        self.guardrails = guardrails

    async def process(
        self, code_patch: str, test_patch: str
    ) -> StructuredResult[ReviewerOutputSchema]:

        combined_code = f"{code_patch}\n\n{test_patch}"

        if self.guardrails:
            run = lambda code: self.guardrails.cache_tool_call(
                "sandbox", {"code": code}, self.sandbox.execute_code
            )
        else:
            run = self.sandbox.execute_code
        # Docker SDK is blocking: keep it off the event loop.
        success, logs = await asyncio.to_thread(run, combined_code)

        prompt = f"""
You are a Reviewer Agent.

Review the code patch, test patch, and sandbox execution logs.

Decide whether to approve the patch or provide feedback.

Code Patch:
{code_patch}

Test Patch:
{test_patch}

Sandbox Output (Success: {success}):
{logs}
"""
        return await complete_structured(
            self.client, [Message("user", prompt)], ReviewerOutputSchema, **LLM_OPTS
        )