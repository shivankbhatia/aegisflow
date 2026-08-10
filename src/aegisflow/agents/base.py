from pydantic import BaseModel, Field

class BaseAgentSchema(BaseModel):
    rationale: str = Field(
        description="Brief explanation supporting the decision."
    )

class TriageOutputSchema(BaseAgentSchema):
    plan: str = Field(description="The outlined plan to solve the issue.")
    required_files: list[str] = Field(description="List of files that likely need to be modified.")

class CoderOutputSchema(BaseAgentSchema):
    code_patch: str = Field(description="The actual code implementation or patch.")
    test_patch: str = Field(description="Unit tests for the implementation.")

class ReviewerOutputSchema(BaseAgentSchema):
    approved: bool = Field(description="Whether the code passes review and tests.")
    feedback: str = Field(description="Specific feedback if rejected, or approval message.")
