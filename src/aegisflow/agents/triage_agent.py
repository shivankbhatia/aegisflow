from ..llm_gateway import LLMClient, Message, StructuredResult, complete_structured
from .base import LLM_OPTS, TriageOutputSchema


class TriageAgent:

    def __init__(self, client: LLMClient):
        self.client = client

    async def process(self, issue_description: str) -> StructuredResult[TriageOutputSchema]:
        prompt = f"""
You are a Triage Agent.

Analyze the following software issue and create a concrete
reproduction and implementation plan.

Identify which files might need to be changed.

Issue:
{issue_description}
"""
        return await complete_structured(
            self.client, [Message("user", prompt)], TriageOutputSchema, **LLM_OPTS
        )