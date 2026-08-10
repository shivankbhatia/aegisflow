from openai.types.beta import chatkit_workflow
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from .guardrails import GuardrailsInterceptor
from ..agents.triage_agent import TriageAgent
from ..agents.coder_agent import CoderAgent
from ..agents.reviewer_agent import ReviewerAgent

# Define the state schema
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]

    task: str

    # Triage output
    plan: str
    required_files: list[str]

    # Coder output
    code_patch: str
    test_patch: str

    # Reviewer output
    approved: bool
    feedback: str

    # Execution
    test_results: str
    errors: str

    recursion_count: int

guardrails = GuardrailsInterceptor()

triage_agent = TriageAgent()
coder_agent = CoderAgent()
reviewer_agent = ReviewerAgent(
    llm_model="openai/gpt-oss-20b"
)

def triage_node(state: AgentState):
    recursion_count = guardrails.check_recursion(state)

    result = triage_agent.process(state["task"])

    return {
        "plan": result.plan,
        "required_files": result.required_files,
        "recursion_count": recursion_count,
    }


def coder_node(state: AgentState):
    recursion_count = guardrails.check_recursion(state)

    result = coder_agent.process(
        plan=state["plan"],
        files=state["required_files"],
    )

    return {
        "code_patch": result.code_patch,
        "test_patch": result.test_patch,
        "recursion_count": recursion_count,
    }


def reviewer_node(state: AgentState):
    recursion_count = guardrails.check_recursion(state)

    result = reviewer_agent.process(
        code_patch=state["code_patch"],
        test_patch=state["test_patch"],
    )

    return {
        "approved": result.approved,
        "feedback": result.feedback,
        "recursion_count": recursion_count,
    }

def build_workflow():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("reviewer", reviewer_node)

    # Add edges
    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "coder")
    workflow.add_edge("coder", "reviewer")


    def review_router(state: AgentState):
        if state["approved"]:
            return "end"

        return "coder"


    workflow.add_conditional_edges(
        "reviewer",
        review_router,
        {
            "coder": "coder",
            "end": END,
        },
    )

    # Note: Checkpointer will be attached when compiling the workflow in main.py
    return workflow.compile()

def run_workflow(task: str):
    workflow = build_workflow()

    return workflow.invoke({
        "task": task,
        "recursion_count": 0,
    })