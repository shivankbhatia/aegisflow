"""Quota-free document-level retrieval baseline for the Phase 3 golden set."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.rag.evaluation import load_questions
from aegisflow.rag.embeddings import MODEL_NAME
from aegisflow.rag.runtime import load_rag_runtime
from aegisflow.rag.manifest import attach_run_manifest


async def run() -> Path:
    load_dotenv()
    questions_path = Path(os.getenv("AEGIS_GOLDEN_SET", "eval/golden.jsonl"))
    questions = load_questions(questions_path)
    expected_count = int(os.getenv("AEGIS_EVAL_EXPECTED_COUNT", "81"))
    expected_multi = int(os.getenv("AEGIS_EVAL_EXPECTED_MULTI_HOP", "15"))
    if len(questions) != expected_count or sum(q.kind == "multi_hop" for q in questions) != expected_multi:
        raise RuntimeError(
            f"Expected {expected_count} questions with {expected_multi} multi-hop cases"
        )
    runtime = await load_rag_runtime(os.environ["DATABASE_URL"])
    rows = []
    for index, q in enumerate(questions, 1):
        started = time.perf_counter()
        chunk_ids = await runtime.retrieve(q.question, 10)
        chunks = await asyncio.to_thread(
            runtime.index.fetch_chunks, chunk_ids, strategy=runtime.strategy,
        )
        retrieved_docs = sorted({chunks[cid]["doc_id"] for cid in chunk_ids if cid in chunks})
        gold_docs = sorted({cid.split("#", 1)[0] for cid in q.gold_chunk_ids})
        rows.append({
            "id": q.id,
            "kind": q.kind,
            "question": q.question,
            "gold_document_ids": gold_docs,
            "retrieved_document_ids": retrieved_docs,
            "retrieved_chunk_ids": chunk_ids,
            "all_gold_documents_retrieved": bool(gold_docs) and set(gold_docs) <= set(retrieved_docs),
            "latency_s": time.perf_counter() - started,
        })
        print(f"[{index}/{len(questions)}] {q.id}: all_gold={rows[-1]['all_gold_documents_retrieved']}")

    multi = [row for row in rows if row["kind"] == "multi_hop"]
    report = attach_run_manifest({
        "name": "phase3-retrieval-only-baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "pipeline": "plain served retriever; document-level coverage; no LLM calls",
        "eval_set": str(questions_path),
        "eval_set_count": len(rows),
        "model": "not applicable (retrieval-only)",
        "retrieval_model": os.getenv("AEGIS_EMBEDDING_MODEL", MODEL_NAME),
        "retrieval_mode": runtime.mode,
        "chunking_strategy": runtime.strategy,
        "k": 10,
        "metrics": {
            "multi_hop_all_gold_documents_retrieved_fraction": (
                sum(row["all_gold_documents_retrieved"] for row in multi) / len(multi)
            ),
            "multi_hop_n": len(multi),
            "all_question_gold_document_coverage_fraction": (
                sum(row["all_gold_documents_retrieved"] for row in rows) / len(rows)
            ),
            "latency_mean_s": sum(row["latency_s"] for row in rows) / len(rows),
        },
        "per_question": rows,
    })
    output = Path(os.getenv("AEGIS_RETRIEVAL_BASELINE_OUTPUT", "eval/results/phase3-retrieval-baseline.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "metrics": report["metrics"]}, indent=2))
    return output


if __name__ == "__main__":
    asyncio.run(run())
