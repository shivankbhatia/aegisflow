from ..llm.client import get_llm
from .base import CoderOutputSchema


class CoderAgent:

    def __init__(self, llm_model: str | None = None):
        self.llm = get_llm(llm_model)

    def process(
        self,
        plan: str,
        files: list[str],
    ) -> CoderOutputSchema:

        prompt = f"""
You are a Coder Agent.

Implement a fix for the given plan.

Also write a unit test to verify the fix.

Plan:
{plan}

Relevant Files:
{', '.join(files)}
"""

        structured_llm = self.llm.with_structured_output(
            CoderOutputSchema
        )

        return structured_llm.invoke(prompt)