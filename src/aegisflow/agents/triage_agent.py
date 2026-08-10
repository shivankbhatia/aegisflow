import os
from ..llm.client import get_llm
from .base import TriageOutputSchema

class TriageAgent:
    def __init__(self, llm_model: str | None = None):
        model = llm_model or os.getenv(
            "AEGIS_MODEL",
            "llama-3.1-8b-instant"
        )

        self.llm = get_llm(model)

    def process(self, issue_description: str) -> TriageOutputSchema:
        prompt = f"""
        You are a Triage Agent.

        Analyze the following software issue and create
        a concrete reproduction and implementation plan.

        Identify which files might need to be changed.

        Issue:
        {issue_description}
        """

        structured_llm = self.llm.with_structured_output(
            TriageOutputSchema
        )

        return structured_llm.invoke(prompt)