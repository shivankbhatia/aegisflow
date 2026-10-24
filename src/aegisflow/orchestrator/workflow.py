from dataclasses import dataclass
from typing import Annotated, Any, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from ..agents.coder_agent import CoderAgent
from ..agents.reviewer_agent import ReviewerAgent
from ..agents.triage_agent import TriageAgent
from ..llm_gateway import from_env
from .guardrails import GuardrailsInterceptor


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    task: str
    plan: str
    required_files: list[str]
    code_patch: str
    test_patch: str
    approved: bool
    feedback: str
    test_results: str
    errors: str
    recursion_count: int  # coder attempts so far
    tokens_used: int  # cumulative, checkpointed, enforced by guardrails


@dataclass
class Agents:
    triage: Any
    coder: Any
    reviewer: Any


def default_agents(guardrails: GuardrailsInterceptor, client=None) -> Agents:
    """Needs GROQ_API_KEY and a running Docker daemon."""
    if client is None:
        load_dotenv()
        client = from_env()
    return Agents(
        TriageAgent(client),
        CoderAgent(client),
        ReviewerAgent(client, guardrails=guardrails),
    )


def review_router(state: AgentState):
    return "end" if state["approved"] else "coder"


def build_workflow(
    agents: Agents | None = None,
    checkpointer=None,
    guardrails: GuardrailsInterceptor | None = None,
):
    guardrails = guardrails or GuardrailsInterceptor()
    agents = agents or default_agents(guardrails)

    def spend(state: AgentState, res) -> int:
        total = state.get("tokens_used", 0) + res.usage.total_tokens
        guardrails.check_budget(total)
        return total

    async def triage_node(state: AgentState):
        res = await agents.triage.process(state["task"])
        return {
            "plan": res.value.plan,
            "required_files": res.value.required_files,
            "tokens_used": spend(state, res),
        }

    async def coder_node(state: AgentState):
        count = guardrails.check_recursion(state)  # counts coder attempts only
        res = await agents.coder.process(
            plan=state["plan"],
            files=state["required_files"],
            feedback=state.get("feedback", ""),
            previous_code=state.get("code_patch", ""),
            previous_tests=state.get("test_patch", ""),
        )
        return {
            "code_patch": res.value.code_patch,
            "test_patch": res.value.test_patch,
            "recursion_count": count,
            "tokens_used": spend(state, res),
        }

    async def reviewer_node(state: AgentState):
        res = await agents.reviewer.process(
            code_patch=state["code_patch"],
            test_patch=state["test_patch"],
        )
        return {
            "approved": res.value.approved,
            "feedback": res.value.feedback,
            "tokens_used": spend(state, res),
        }

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


async def run_workflow(
    task: str,
    thread_id: str = "default",
    agents: Agents | None = None,
    checkpointer=None,
    guardrails: GuardrailsInterceptor | None = None,
):
    graph = build_workflow(agents, checkpointer, guardrails)
    config = {"configurable": {"thread_id": thread_id}}
    return await graph.ainvoke(
        {"task": task, "recursion_count": 0, "tokens_used": 0}, config
    )


async def resume_workflow(
    thread_id: str,
    checkpointer,
    agents: Agents | None = None,
    guardrails: GuardrailsInterceptor | None = None,
):
    """Resume after a crash: input=None continues from the last checkpoint."""
    graph = build_workflow(agents, checkpointer, guardrails)
    config = {"configurable": {"thread_id": thread_id}}
    return await graph.ainvoke(None, config)