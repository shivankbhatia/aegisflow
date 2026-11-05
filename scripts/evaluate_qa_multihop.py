"""Planner-only multi-hop retrieval eval (no answer synthesis)."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.llm_gateway import Message, from_env
from aegisflow.rag.evaluation import load_questions, paired_bootstrap
from aegisflow.rag.qa_workflow import _PLANNER_PROMPT, build_qa_workflow, make_retrieval_tool
from aegisflow.rag.runtime import load_rag_runtime
from aegisflow.rag.manifest import attach_run_manifest


async def run() -> Path:
    load_dotenv()
    baseline_path = Path(os.getenv("AEGIS_PHASE3_RETRIEVAL_BASELINE", "eval/results/phase3-retrieval-baseline.json"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not baseline.get("complete"):
        raise RuntimeError("A complete retrieval-only baseline is required before comparison")
    baseline_multi = {row["id"] for row in baseline["per_question"] if row["kind"] == "multi_hop"}
    if not baseline_multi:
        raise RuntimeError("Baseline must contain multi-hop questions")

    question_path = os.getenv("AEGIS_GOLDEN_SET", "eval/golden.jsonl")
    questions = [q for q in load_questions(question_path) if q.id in baseline_multi]
    if {q.id for q in questions} != baseline_multi:
        raise RuntimeError("Question set and retrieval baseline IDs do not match")
    runtime = await load_rag_runtime(os.environ["DATABASE_URL"])
    if runtime.mode != baseline.get("retrieval_mode") or runtime.strategy != baseline.get("chunking_strategy"):
        raise RuntimeError("Planner comparison must use the same retrieval mode and chunking strategy as baseline")
    retriever = make_retrieval_tool(runtime, k=10, max_tokens=3000)
    client = from_env(max_concurrency=1)
    per_question_prompt_tokens = [
        client.count_tokens([
            Message("system", _PLANNER_PROMPT),
            Message("user", question.question),
        ])
        for question in questions
    ]
    planned_token_ceiling = sum(per_question_prompt_tokens) + 400 * len(questions)
    print(json.dumps({
        "estimate_before_run": {
            "planner_calls": len(questions),
            "estimated_input_tokens": sum(per_question_prompt_tokens),
            "max_output_tokens_per_call": 400,
            "estimated_total_tokens_excluding_provider_overhead": planned_token_ceiling,
            "basis": "local prompt token count plus configured output cap; excludes provider overhead",
            "model": client.model,
        }
    }, indent=2))
    rows = []
    try:
        graph = build_qa_workflow(
            client, retriever, planner_retrieval_only=True,
        )
        for index, question in enumerate(questions, 1):
            result = await graph.ainvoke({
                "question": question.question, "tokens_used": 0,
                "attempts": 0, "branch_results": [],
            })
            sub_questions = result["sub_questions"]
            branch_evidence = result["evidence"]
            gold_docs = sorted({cid.split("#", 1)[0] for cid in question.gold_chunk_ids})
            retrieved_docs = sorted({item["chunk_id"].split("#", 1)[0] for item in branch_evidence})
            rows.append({
                "id": question.id,
                "kind": question.kind,
                "sub_questions": sub_questions,
                "gold_document_ids": gold_docs,
                "retrieved_document_ids": retrieved_docs,
                "all_gold_documents_retrieved": bool(gold_docs) and set(gold_docs) <= set(retrieved_docs),
                "planner_tokens": result["tokens_used"],
                "planner_attempts": result["attempts"],
            })
            print(f"[{index}/15] {question.id}: all_gold={rows[-1]['all_gold_documents_retrieved']}")
    finally:
        await client.aclose()

    report = attach_run_manifest({
        "name": "qa-planner-only-multihop-document-coverage",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_report": str(baseline_path),
        "model": client.model,
        "judge_status": "No answer judge; document coverage is deterministic",
        "pipeline": "planner plus fan-out retrieval; no synthesis or verification",
        "eval_set": question_path,
        "n": len(rows),
        "all_gold_documents_retrieved_rate": sum(row["all_gold_documents_retrieved"] for row in rows) / len(rows),
        "per_question": rows,
    })
    baseline_multi_report = {
        **baseline,
        "per_question": [row for row in baseline["per_question"] if row["kind"] == "multi_hop"],
    }
    report["paired_bootstrap_vs_baseline"] = paired_bootstrap(
        baseline_multi_report, report, "all_gold_documents_retrieved", resamples=10_000, seed=0,
    )
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"qa-multihop-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"rate": report["all_gold_documents_retrieved_rate"],
                      "paired_bootstrap_vs_baseline": report["paired_bootstrap_vs_baseline"]}, indent=2))
    return output


if __name__ == "__main__":
    asyncio.run(run())
