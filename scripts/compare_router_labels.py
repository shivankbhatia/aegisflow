"""Print differences between the independent route labels and Codex labels."""

import json
from pathlib import Path


def read_jsonl(path: str) -> dict[str, dict]:
    return {
        row["id"]: row
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def main() -> None:
    independent = read_jsonl("eval/router_relabels_second_pass.jsonl")
    codex = read_jsonl("eval/router_queries.jsonl")
    if independent.keys() != codex.keys():
        raise SystemExit("independent and Codex label files have different query IDs")
    differences = [
        {
            "id": case_id,
            "independent_label": independent[case_id]["label"],
            "codex_label": codex[case_id].get("route", codex[case_id].get("label")),
            "question": codex[case_id].get("question"),
            "insight": independent[case_id]["rationale"],
        }
        for case_id in sorted(independent)
        if independent[case_id]["label"]
        != codex[case_id].get("route", codex[case_id].get("label"))
    ]
    print(json.dumps({"n": len(independent), "disagreement_count": len(differences), "disagreements": differences}, indent=2))


if __name__ == "__main__":
    main()
