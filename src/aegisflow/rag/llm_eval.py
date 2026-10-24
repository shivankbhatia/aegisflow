"""LLM-assisted eval pieces: golden-question drafting and a faithfulness judge."""
import asyncio
from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel, Field

from ..llm_gateway import LLMClient, Message, StructuredOutputError, complete_structured
from .evaluation import GoldenQuestion
from .types import Chunk

# reasoning_effort is gpt-oss specific; drop it if you switch models.
GEN_OPTS = {"max_tokens": 1500, "reasoning_effort": "low"}


class GeneratedQuestion(BaseModel):
    question: str = Field(description="Self-contained question; never mentions 'the passage' or 'the text'.")
    reference_answer: str = Field(description="Short answer, fully supported by the passages.")


async def generate_questions(
    client: LLMClient,
    groups: Sequence[Sequence[Chunk]],
    kind: str = "single",
    *,
    concurrency: int = 5,
    id_prefix: str = "q",
    opts: dict | None = None,
) -> list[GoldenQuestion]:
    """One question per group of chunks. kind='multi_hop' requires every passage.
    Groups the model cannot structure are skipped. Output is unverified."""
    opts, sem = opts or GEN_OPTS, asyncio.Semaphore(concurrency)
    need = (
        "Write ONE question that can only be answered by combining ALL the passages."
        if kind == "multi_hop"
        else "Write ONE question that can be answered from this passage alone."
    )

    async def one(i: int, group: Sequence[Chunk]) -> GoldenQuestion | None:
        passages = "\n\n".join(f"[Passage {j + 1}]\n{c.text}" for j, c in enumerate(group))
        prompt = (
            f"{need}\nThe question must make sense to someone who has not seen the passages "
            f"(name the company, product or topic explicitly). Also give a short reference answer.\n\n{passages}"
        )
        async with sem:
            try:
                res = await complete_structured(client, [Message("user", prompt)], GeneratedQuestion, **opts)
            except StructuredOutputError:
                return None
        return GoldenQuestion(
            id=f"{id_prefix}{i:03d}",
            question=res.value.question.strip(),
            gold_chunk_ids=tuple(c.id for c in group),
            kind=kind,
            reference_answer=res.value.reference_answer.strip(),
        )

    out = await asyncio.gather(*[one(i, g) for i, g in enumerate(groups)])
    return [q for q in out if q is not None]


class FaithfulnessVerdict(BaseModel):
    supported_claims: int = Field(ge=0, description="Claims in the answer fully supported by the context.")
    unsupported_claims: int = Field(ge=0, description="Claims not supported by, or contradicting, the context.")
    unsupported: list[str] = Field(default_factory=list, description="Each unsupported claim, quoted briefly.")


@dataclass(frozen=True)
class Judgement:
    score: float  # supported / total claims; 1.0 when the answer makes no claims
    verdict: FaithfulnessVerdict
    tokens_used: int = 0
    attempts: int = 0


async def judge_faithfulness(
    client: LLMClient,
    question: str,
    answer: str,
    contexts: Sequence[str],
    opts: dict | None = None,
) -> Judgement:
    ctx = "\n\n".join(f"[Context {i + 1}]\n{c}" for i, c in enumerate(contexts))
    prompt = (
        "You are a strict fact-checker. Split the ANSWER into atomic factual claims and count how many "
        "are fully supported by the CONTEXT and how many are not. Use only the context, not world knowledge.\n\n"
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n{ctx}"
    )
    res = await complete_structured(client, [Message("user", prompt)], FaithfulnessVerdict, **(opts or GEN_OPTS))
    v = res.value
    total = v.supported_claims + v.unsupported_claims
    return Judgement(v.supported_claims / total if total else 1.0, v, res.usage.total_tokens, res.attempts)
