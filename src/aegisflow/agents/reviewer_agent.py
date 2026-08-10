from langchain_openai import ChatOpenAI
from .base import ReviewerOutputSchema
from ..llm.client import get_llm
from ..sandbox.executor import SandboxExecutor

class ReviewerAgent:
    def __init__(self, llm_model: str):
        self.llm = get_llm(llm_model)
        self.sandbox = SandboxExecutor()

    def process(self, code_patch: str, test_patch: str) -> ReviewerOutputSchema:
        # 1. Run the code and tests in the sandbox
        combined_code = f"{code_patch}\n\n{test_patch}"
        success, logs = self.sandbox.execute_code(combined_code)

        # 2. Have the LLM review the results
        prompt = f"""
        You are a Reviewer Agent. Review the code patch, test patch, and sandbox execution logs.
        Decide whether to approve the patch or provide feedback.

        Code Patch:
        {code_patch}

        Test Patch:
        {test_patch}

        Sandbox Output (Success: {success}):
        {logs}
        """

        structured_llm = self.llm.with_structured_output(ReviewerOutputSchema)
        return structured_llm.invoke(prompt)
