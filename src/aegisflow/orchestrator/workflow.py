from dataclasses import dataclass
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from ..agents.coder_agent import CoderAgent
from ..agents.reviewer_agent import ReviewerAgent
from ..agents.triage_agent import TriageAgent
from .guardrails import GuardrailsInterceptor


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    task: str

    # Triage
    plan: str
    required_files: list[str]

    # Coder
    code_patch: str
    test_patch: str

    # Reviewer
    approved: bool
    feedback: str

    # Sandbox
    test_results: str
    errors: str

    # Loop protection: number of coder attempts so far
    recursion_count: int


@dataclass
class Agents:
    triage: Any
    coder: Any
    reviewer: Any


def default_agents() -> Agents:
    """Built lazily: needs GROQ_API_KEY and a running Docker daemon."""
    return Agents(TriageAgent(), CoderAgent(), ReviewerAgent())


def review_router(state: AgentState):
    return "end" if state["approved"] else "coder"


def build_workflow(
    agents: Agents | None = None,
    checkpointer=None,
    guardrails: GuardrailsInterceptor | None = None,
):
    agents = agents or default_agents()
    guardrails = guardrails or GuardrailsInterceptor()

    def triage_node(state: AgentState):
        result = agents.triage.process(state["task"])
        return {"plan": result.plan, "required_files": result.required_files}

    def coder_node(state: AgentState):
        # Circuit breaker counts coder attempts only (the retry loop).
        recursion_count = guardrails.check_recursion(state)
        result = agents.coder.process(
            plan=state["plan"],
            files=state["required_files"],
        )
        return {
            "code_patch": result.code_patch,
            "test_patch": result.test_patch,
            "recursion_count": recursion_count,
        }

    def reviewer_node(state: AgentState):
        result = agents.reviewer.process(
            code_patch=state["code_patch"],
            test_patch=state["test_patch"],
        )
        return {"approved": result.approved, "feedback": result.feedback}

    workflow = StateGraph(AgentState)
    workflow.add_node("triage", triage_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "coder")
    workflow.add_edge("coder", "reviewer")
    workflow.add_conditional_edges(
        "reviewer", review_router, {"end": END, "coder": "coder"}
    )

    return workflow.compile(checkpointer=checkpointer)


def run_workflow(
    task: str,
    thread_id: str = "default",
    agents: Agents | None = None,
    checkpointer=None,
):
    graph = build_workflow(agents=agents, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"task": task, "recursion_count": 0}, config)


def resume_workflow(
    thread_id: str,
    checkpointer,
    agents: Agents | None = None,
):
    """Resume after a crash. input=None continues from the last checkpoint."""
    graph = build_workflow(agents=agents, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(None, config)
