"""Compact checkpoint state for the documentation QA graph."""

from operator import add
from typing import Annotated, TypedDict


class Evidence(TypedDict):
    chunk_id: str
    source: str
    heading_path: list[str]
    text: str  # trimmed retrieval text; never persist full corpus chunks


class Citation(TypedDict):
    chunk_id: str
    marker: str


class BranchResult(TypedDict):
    branch_index: int
    sub_question: str
    evidence: list[Evidence]


class QAState(TypedDict, total=False):
    question: str
    route: str
    sub_questions: list[str]
    evidence: list[Evidence]
    draft_answer: str
    citations: list[Citation]
    verdict: str
    attempts: int
    tokens_used: int
    final_answer: str
    sub_question: str
    branch_index: int
    branch_results: Annotated[list[BranchResult], add]
    revision_count: int
    verify_feedback: str
    faithfulness_score: float
    citation_valid: bool
    guardrail_reason: str
    approval_required: bool
    approval_decision: bool
    approval_feedback: str
    verification_error: str
    output_gate_error: str
    projected_token_cost: int
