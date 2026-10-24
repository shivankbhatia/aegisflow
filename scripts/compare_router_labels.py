"""Compare neutral-ID router labels with the original Codex labels."""

import json
from collections import Counter
from pathlib import Path

from aegisflow.rag.manifest import attach_run_manifest


def read_rows(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_jsonl(path: str) -> dict[str, dict]:
    return {row["id"]: row for row in read_rows(path)}


def main() -> None:
    blind = read_jsonl("eval/router_blind_labels.jsonl")
    mapping = {
        row["blind_id"]: row["source_id"]
        for row in read_rows("eval/router_blind_id_map.jsonl")
    }
    codex = read_jsonl("eval/router_queries.jsonl")
    if blind.keys() != mapping.keys() or set(mapping.values()) != codex.keys():
        raise SystemExit("blind labels, ID map, and Codex labels do not have matching query IDs")
    disagreements = []
    aligned = []
    for blind_id in sorted(blind):
        source_id = mapping[blind_id]
        row = {
            "blind_id": blind_id,
            "source_id": source_id,
            "blind_label": blind[blind_id]["label"],
            "codex_label": codex[source_id]["route"],
            "question": codex[source_id]["question"],
            "rationale": blind[blind_id]["rationale"],
        }
        aligned.append(row)
        if row["blind_label"] != row["codex_label"]:
            disagreements.append(row)
    report = attach_run_manifest({
        "name": "router-blind-relabel-v1",
        "n": len(aligned),
        "authorship": "Assistant relabel from shuffled neutral-ID prompts; no 24-hour washout",
        "accuracy_vs_codex": (len(aligned) - len(disagreements)) / len(aligned),
        "counts_by_label": dict(Counter(row["blind_label"] for row in aligned)),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "per_query": aligned,
    })
    out = Path("eval/results/router-blind-relabel.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "n": report["n"],
                      "accuracy_vs_codex": report["accuracy_vs_codex"],
                      "disagreement_count": report["disagreement_count"],
                      "counts_by_label": report["counts_by_label"]}, indent=2))


if __name__ == "__main__":
    main()
