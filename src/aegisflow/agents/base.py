from pydantic import BaseModel, Field

class BaseAgentSchema(BaseModel):
    rationale: str = Field(
        description="Brief explanation supporting the decision."
    )

class TriageOutputSchema(BaseAgentSchema):
    plan: str = Field(description="The outlined plan to solve the issue.")
    required_files: list[str] = Field(description="List of files that likely need to be modified.")

class CoderOutputSchema(BaseAgentSchema):
    code_patch: str = Field(description="Complete, self-contained, runnable Python source. NOT a diff. No markdown fences.")
    test_patch: str = Field(description="Plain Python assert statements that exercise code_patch. No imports from files, no pytest, no markdown fences.")
    
class ReviewerOutputSchema(BaseAgentSchema):
    approved: bool = Field(description="Whether the code passes review and tests.")
    feedback: str = Field(description="Specific feedback if rejected, or approval message.")


# gpt-oss is a reasoning model: reasoning tokens count against max_tokens.
# reasoning_effort is gpt-oss specific; drop it if you switch models.
LLM_OPTS = {"max_tokens": 2000, "reasoning_effort": "low"}
