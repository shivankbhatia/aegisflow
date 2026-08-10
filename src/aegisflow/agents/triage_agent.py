from langchain_openai import ChatOpenAI
from .base import TriageOutputSchema

import os

class TriageAgent:
    def __init__(self, llm_model: str | None = None):
        model = llm_model or os.getenv("AEGIS_MODEL", "llama-3.1-70b-versatile")
        self.llm = ChatOpenAI(model=model)

    def process(self, issue_description: str) -> TriageOutputSchema:
        prompt = f"""
        You are a Triage Agent. Analyze the following issue and create a reproduction plan.
        Identify which files might need to be changed.
        
        Issue:
        {issue_description}
        """
        
        # Enforce Pydantic schema on output
        structured_llm = self.llm.with_structured_output(TriageOutputSchema)
        return structured_llm.invoke(prompt)
