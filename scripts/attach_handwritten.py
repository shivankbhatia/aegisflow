"""Attach user-authored questions to their nearest indexed source passages."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.rag.embeddings import BGEEmbedder
from aegisflow.rag.evaluation import GoldenQuestion, load_questions, save_questions
from aegisflow.rag.vector_index import PgVectorIndex


def main() -> None:
    load_dotenv()
    dsn = os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")
    index = PgVectorIndex(dsn)
    embedder = BGEEmbedder()
    drafts_path = Path("eval/golden.jsonl")
    draft_questions = load_questions(drafts_path) if drafts_path.exists() else []
    by_id = {question.id: question for question in draft_questions}
    added = []
    for line in Path("eval/handwritten_questions.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        vector = embedder.embed_query(item["question"])
        hits = index.search(vector, strategy="structure-aware", source=item["source"], k=1)
        if not hits:
            raise RuntimeError(f"No supporting chunk found for {item['id']} in {item['source']}")
        added.append(
            GoldenQuestion(
                id=item["id"],
                question=item["question"],
                gold_chunk_ids=(hits[0][0],),
                reference_answer=None,
                verified=False,
            )
        )
    combined = [q for q in draft_questions if q.id not in {x.id for x in added}] + added
    save_questions(combined, drafts_path)
    print(f"Attached {len(added)} hand-written questions to {drafts_path}; all remain unverified")
    for question in added:
        print(f"{question.id}: {question.gold_chunk_ids[0]}")


if __name__ == "__main__":
    main()
