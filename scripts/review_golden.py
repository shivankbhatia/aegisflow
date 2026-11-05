"""Interactively review eval/golden.jsonl and mark accepted questions verified."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.llm_gateway import from_env
from aegisflow.rag.evaluation import (
    load_questions,
    review_questions,
    save_questions,
)
from aegisflow.rag.vector_index import PgVectorIndex

from draft_golden import STRATEGY, load_indexed_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--write", action="store_true", help="save kept/edited questions to the input path")
    args = parser.parse_args()
    load_dotenv()
    dsn = os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")
    index = PgVectorIndex(dsn)
    chunks = load_indexed_chunks(index)
    by_id = {chunk.id: chunk for chunk in chunks}
    output_path = args.questions or Path("eval/golden.jsonl")
    input_paths = (
        [args.questions]
        if args.questions
        else [Path("eval/golden.jsonl"), Path("eval/golden.pending.jsonl")]
    )
    drafts = []
    seen_ids = set()
    for input_path in input_paths:
        if not input_path.exists():
            continue
        for question in load_questions(input_path):
            if question.id not in seen_ids:
                drafts.append(question)
                seen_ids.add(question.id)
    if not drafts:
        raise SystemExit("No questions found to review")
    reviewed = review_questions(drafts, by_id)
    if args.write:
        verified = [question for question in reviewed if question.verified]
        pending = [question for question in reviewed if not question.verified]
        pending_path = output_path.with_name(f"{output_path.stem}.pending{output_path.suffix}")
        save_questions(verified, output_path)
        save_questions(pending, pending_path)
        print(f"Saved {len(verified)} verified questions to {output_path}")
        print(f"Saved {len(pending)} questions still needing review to {pending_path}")
    else:
        print(f"Review session finished: {sum(q.verified for q in reviewed)} verified; run again with --write to save")


if __name__ == "__main__":
    main()
