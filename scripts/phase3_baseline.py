"""Run the plain /query baseline over the frozen Phase 3 development set."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from aegisflow.llm_gateway import Message, from_env, complete_structured
from aegisflow.rag.evaluation import load_questions
from aegisflow.rag.llm_eval import judge_faithfulness
from aegisflow.rag.runtime import load_rag_runtime
from aegisflow.rag.manifest import attach_run_manifest


class CorrectnessVerdict(BaseModel):
    score: float = Field(ge=0, le=1, description="How completely the response answers the question correctly")
    rationale: str


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


async def correctness_judge(client, question: str, reference: str, answer: str) -> dict:
    prompt = (
        "Compare the answer with the reference for factual correctness and completeness. "
        "Ignore wording differences. Give 1 for fully correct, 0 for incorrect, and an "
        "intermediate score for partial correctness. Do not reward unsupported extra claims.\n\n"
        f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nANSWER:\n{answer}"
    )
    result = await complete_structured(
        client, [Message("user", prompt)], CorrectnessVerdict,
        max_tokens=500, reasoning_effort="low",
    )
    return {**result.value.model_dump(), "tokens": result.usage.total_tokens, "attempts": result.attempts}


async def query_once(http, url: str, question: str, k: int) -> dict:
    started = time.perf_counter()
    answer: list[str] = []
    done: dict = {}
    async with http.stream("POST", f"{url}/query", json={"query": question, "k": k}) as response:
        response.raise_for_status()
        event = None
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event:
                data = json.loads(line[5:].strip())
                if event == "token":
                    answer.append(data.get("text", ""))
                elif event == "done":
                    done = data
                elif event == "error":
                    raise RuntimeError(data.get("detail", "query failed"))
                event = None
    return {"answer": "".join(answer), "tokens": done.get("tokens"),
            "latency_s": (time.perf_counter() - started),
            "timings_ms": done.get("timings_ms", {})}


async def run() -> Path:
    load_dotenv()
    questions = load_questions(os.getenv("AEGIS_GOLDEN_SET", "eval/golden.jsonl"))
    expected_count = int(os.getenv("AEGIS_EVAL_EXPECTED_COUNT", "81"))
    if len(questions) != expected_count:
        raise RuntimeError(f"Expected {expected_count} questions; got {len(questions)}")
    if sum(q.kind == "multi_hop" for q in questions) != 15:
        raise RuntimeError("Expected 15 multi-hop questions in the frozen dev set")

    api_url = os.getenv("AEGIS_API_URL", "http://127.0.0.1:8000").rstrip("/")
    timeout = httpx.Timeout(float(os.getenv("AEGIS_EVAL_TIMEOUT_S", "180")))
    client = from_env(max_concurrency=1)
    runtime = await load_rag_runtime(os.environ["DATABASE_URL"])
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / os.getenv("AEGIS_PHASE3_BASELINE_OUTPUT", "phase3-baseline.json")
    rows = json.loads(output.read_text(encoding="utf-8")).get("per_question", []) if output.exists() else []
    rows_by_id = {row["id"]: row for row in rows}
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            for i, q in enumerate(questions, 1):
                row = rows_by_id.get(q.id)
                if row is None or "answer" not in row:
                    # Mirror the live endpoint's retrieval result set and k.
                    retrieved = await runtime.retrieve(q.question, 10)
                    result = await query_once(http, api_url, q.question, 10)
                    evidence = await asyncio.to_thread(
                        runtime.index.fetch_chunks, retrieved,
                        strategy=runtime.strategy,
                    )
                    retrieved_docs = sorted({evidence[cid]["doc_id"] for cid in retrieved if cid in evidence})
                    gold_docs = sorted({cid.split("#", 1)[0] for cid in q.gold_chunk_ids})
                    row = {
                        "id": q.id, "kind": q.kind, "question": q.question,
                        "reference_answer": q.reference_answer,
                        "answer": result["answer"], "retrieved_chunk_ids": retrieved,
                        "retrieved_document_ids": retrieved_docs, "gold_document_ids": gold_docs,
                        "all_gold_documents_retrieved": bool(gold_docs) and set(gold_docs) <= set(retrieved_docs),
                        "correctness": None, "correctness_rationale": None,
                        "correctness_judge_tokens": None,
                        "faithfulness": None, "faithfulness_supported_claims": None,
                        "faithfulness_judge_tokens": None,
                        "faithfulness_unsupported_claims": None, "faithfulness_unsupported": [],
                        "tokens": result["tokens"], "latency_s": result["latency_s"],
                        "timings_ms": result["timings_ms"], "human_correctness": None,
                        "human_faithfulness": None,
                    }
                    rows.append(row)
                    rows_by_id[q.id] = row
                    save_checkpoint(rows, output, api_url)
                if row["faithfulness"] is None:
                    evidence = await asyncio.to_thread(
                        runtime.index.fetch_chunks, row["retrieved_chunk_ids"],
                        strategy=runtime.strategy,
                    )
                    faith = await judge_faithfulness(
                        client, q.question, row["answer"],
                        [evidence[cid]["text"] for cid in row["retrieved_chunk_ids"] if cid in evidence],
                    )
                    row.update({
                        "faithfulness": faith.score,
                        "faithfulness_judge_tokens": faith.tokens_used,
                        "faithfulness_judge_attempts": faith.attempts,
                        "faithfulness_supported_claims": faith.verdict.supported_claims,
                        "faithfulness_unsupported_claims": faith.verdict.unsupported_claims,
                        "faithfulness_unsupported": faith.verdict.unsupported,
                    })
                    save_checkpoint(rows, output, api_url)
                if row["correctness"] is None:
                    correct = await correctness_judge(client, q.question, q.reference_answer or "", row["answer"])
                    row.update({
                        "correctness": correct["score"],
                        "correctness_rationale": correct["rationale"],
                        "correctness_judge_tokens": correct["tokens"],
                        "correctness_judge_attempts": correct["attempts"],
                    })
                    save_checkpoint(rows, output, api_url)
                print(f"[{i}/{len(questions)}] {q.id} judges done")
    finally:
        await client.aclose()

    report = make_report(rows, api_url, complete=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # A balanced, deterministic 15-answer packet for the requested human calibration.
    selected = sorted(rows, key=lambda r: r["id"])[:15]
    review_path = output_dir / "phase3-human-review.jsonl"
    review_path.write_text("".join(json.dumps({
        "id": r["id"], "question": r["question"], "reference_answer": r["reference_answer"],
        "answer": r["answer"], "judge_correctness": r["correctness"],
        "judge_faithfulness": r["faithfulness"],
        "human_correctness": None, "human_faithfulness": None,
    }) + "\n" for r in selected), encoding="utf-8")
    print(json.dumps({"report": str(output), "human_review": str(review_path), "metrics": report["metrics"]}, indent=2))
    return output


def make_report(rows: list[dict], api_url: str, *, complete: bool) -> dict:
    multi = [r for r in rows if r["kind"] == "multi_hop" and r.get("all_gold_documents_retrieved") is not None]
    judged_correctness = [r["correctness"] for r in rows if r.get("correctness") is not None]
    judged_faithfulness = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
    measured = [r for r in rows if r.get("tokens") is not None]
    return attach_run_manifest({
        "name": "phase3-plain-query-baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "pipeline": "plain /query; no QA graph or agent routing",
        "api_url": api_url,
        "eval_set": os.getenv("AEGIS_GOLDEN_SET", "eval/golden.jsonl"),
        "eval_set_target_count": int(os.getenv("AEGIS_EVAL_EXPECTED_COUNT", "81")),
        "model": os.getenv("AEGIS_MODEL", "openai/gpt-oss-20b"),
        "generation_tokens_metric": "plain /query tokens; judge tokens are reported separately",
        "eval_set_count": len(rows),
        "eval_set_split": {"multi_hop": sum(r["kind"] == "multi_hop" for r in rows),
                           "single_hop": sum(r["kind"] == "single" for r in rows)},
        "judge_status": "LLM-judged, uncalibrated",
        "human_labels_completed": 0,
        "metrics": {
            "multi_hop_all_gold_documents_retrieved_fraction": (
                sum(r["all_gold_documents_retrieved"] for r in multi) / len(multi) if multi else None
            ),
            "answer_correctness_mean": sum(judged_correctness) / len(judged_correctness) if judged_correctness else None,
            "answer_correctness_n": len(judged_correctness),
            "faithfulness_mean": sum(judged_faithfulness) / len(judged_faithfulness) if judged_faithfulness else None,
            "faithfulness_n": len(judged_faithfulness),
            "tokens_mean": sum(r["tokens"] for r in measured) / len(measured) if measured else None,
            "latency_p50_s": percentile([r["latency_s"] for r in measured], .5),
            "latency_p95_s": percentile([r["latency_s"] for r in measured], .95),
        },
        "per_question": rows,
    })


def save_checkpoint(rows: list[dict], output: Path, api_url: str) -> None:
    output.write_text(json.dumps(make_report(rows, api_url, complete=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(run())
