from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph, StateGraph, END
from .guardrails import GuardrailsInterceptor

# Define the state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    task: str
    code: str
    test_results: str
    errors: str

guardrails = GuardrailsInterceptor()

def triage_node(state: AgentState):
    # Simulated triage node
    guardrails.check_recursion(state)
    return {"messages": [{"role": "assistant", "content": "Triage complete"}]}

def coder_node(state: AgentState):
    # Simulated coder node
    guardrails.check_recursion(state)
    return {"messages": [{"role": "assistant", "content": "Code generated"}], "code": "def hello(): pass"}

def reviewer_node(state: AgentState):
    # Simulated reviewer node
    guardrails.check_recursion(state)
    # Simulate tests passing
    return {"messages": [{"role": "assistant", "content": "Tests passed"}], "test_results": "OK"}

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
    workflow.add_edge("reviewer", END)

    # Note: Checkpointer will be attached when compiling the workflow in main.py
    return workflow.compile()
