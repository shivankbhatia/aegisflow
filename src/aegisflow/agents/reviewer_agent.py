from ..llm.client import get_llm
from ..sandbox.executor import SandboxExecutor
from .base import ReviewerOutputSchema


class ReviewerAgent:

    def __init__(self, llm_model: str | None = None):
        self.llm = get_llm(llm_model)
        self.sandbox = SandboxExecutor()

    def process(
        self,
        code_patch: str,
        test_patch: str,
    ) -> ReviewerOutputSchema:

        combined_code = f"{code_patch}\n\n{test_patch}"

        success, logs = self.sandbox.execute_code(
            combined_code
        )

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

        structured_llm = self.llm.with_structured_output(
            ReviewerOutputSchema
        )

        return structured_llm.invoke(prompt)