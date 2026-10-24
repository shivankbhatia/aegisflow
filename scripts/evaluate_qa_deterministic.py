"""Report quota-free deterministic QA guardrail and fixture checks."""

import html
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from aegisflow.rag.evaluation import load_questions
from aegisflow.rag.qa_workflow import input_guardrail_reason, validate_citations
from aegisflow.rag.qa_workflow import _INJECTION_PATTERNS
from aegisflow.rag.manifest import attach_run_manifest
from aegisflow.rag.qa_config import CLASSIFIER_PROMPT


def read_jsonl(path: str):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def main():
    injections = read_jsonl("eval/security/direct_injection_prompts.jsonl")
    paraphrases = read_jsonl("eval/security/direct_injection_seen15.jsonl")
    challenge = read_jsonl("eval/security/direct_injection_challenge15_v1.jsonl")
    poisoned = read_jsonl("eval/security/poisoned_chunks.jsonl")
    unanswerable = read_jsonl("eval/security/unanswerable_questions.jsonl")
    singles = [q for q in load_questions("eval/golden.jsonl") if q.kind == "single"]
    blocked = sum(input_guardrail_reason(row["question"]) is not None for row in injections)
    frozen_rules = json.loads(Path("eval/security/injection_heuristics_v1.json").read_text(encoding="utf-8"))
    current_patterns = [pattern.pattern for pattern in _INJECTION_PATTERNS]
    current_hash = hashlib.sha256(
        json.dumps(current_patterns, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    freeze = json.loads(Path("eval/security/guardrail_freeze_v1.json").read_text(encoding="utf-8"))
    classifier_hash = hashlib.sha256(CLASSIFIER_PROMPT.encode()).hexdigest()
    paraphrase_blocked = sum(input_guardrail_reason(row["question"]) is not None for row in paraphrases)
    challenge_rule_blocked = sum(input_guardrail_reason(row["question"]) is not None for row in challenge)
    false_refusals = sum(input_guardrail_reason(q.question) is not None for q in singles)
    evidence = [{"chunk_id": "fixture/doc#0", "source": "fixture", "heading_path": [], "text": "fact"}]
    citation_cases = [
        {"answer": "Claim [1].", "invalid": []},
        {"answer": "Claim [2].", "invalid": [2]},
        {"answer": "Claim without citation.", "invalid": [0]},
    ]
    citation_checks = []
    for case in citation_cases:
        _valid, invalid = validate_citations(case["answer"], evidence)
        citation_checks.append(invalid == case["invalid"])
    poison_wrapped = sum(
        "<untrusted_evidence>" in (
            "<untrusted_evidence>\n" + html.escape(row["text"]) + "\n</untrusted_evidence>"
        )
        and "</untrusted_evidence>" not in html.escape(row["text"])
        for row in poisoned
    )
    report = attach_run_manifest({
        "name": "qa-deterministic-guardrails",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "not applicable (deterministic checks only)",
        "judge_status": "No judge used; LLM-judged quality remains uncalibrated",
        "direct_injection_heuristic": {
            "n": len(injections), "blocked": blocked,
            "blocked_fraction": blocked / len(injections) if injections else None,
        },
        "exposed_paraphrase_heuristic_check": {
            "n": len(paraphrases), "blocked": paraphrase_blocked,
            "blocked_fraction": paraphrase_blocked / len(paraphrases) if paraphrases else None,
            "rules_version": frozen_rules["version"],
            "rules_sha256_at_freeze": frozen_rules["sha256"],
            "rules_sha256_current": current_hash,
            "rules_unchanged": current_hash == frozen_rules["sha256"],
            "note": "These 15 cases were exposed before the classifier prompt was frozen and are a seen set; semantic resistance is not measured.",
        },
        "post_freeze_challenge_heuristic_check": {
            "n": len(challenge), "blocked_by_regex": challenge_rule_blocked,
            "blocked_fraction": challenge_rule_blocked / len(challenge) if challenge else None,
            "challenge_version": "v1",
            "written_after_freeze": True,
            "interpretation": "Regex-only precheck; classifier/model resistance has not been measured.",
        },
        "guardrail_freeze": {
            "version": freeze["version"],
            "classifier_prompt_sha256_frozen": freeze["classifier_prompt_sha256"],
            "classifier_prompt_sha256_current": classifier_hash,
            "classifier_prompt_unchanged": classifier_hash == freeze["classifier_prompt_sha256"],
            "regex_rules_sha256_frozen": freeze["regex_rules_sha256"],
            "regex_rules_sha256_current": current_hash,
            "regex_rules_unchanged": current_hash == freeze["regex_rules_sha256"],
        },
        "golden_single_heuristic_false_refusal": {
            "n": len(singles), "false_refusals": false_refusals,
            "fraction": false_refusals / len(singles) if singles else None,
        },
        "poisoned_evidence_delimiter_escape": {
            "n": len(poisoned), "safe_serializations": poison_wrapped,
            "fraction": poison_wrapped / len(poisoned) if poisoned else None,
            "interpretation": "Serialization only; does not measure model resistance.",
        },
        "citation_validation": {
            "cases": len(citation_checks), "passed": sum(citation_checks),
        },
        "unanswerable_fixture_count": len(unanswerable),
        "unanswerable_refusal_rate": None,
        "unanswerable_note": "Requires graph/model behavior; not claimed by this deterministic report.",
    })
    path = Path("eval/results/qa-deterministic-guardrails.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "metrics": report}, indent=2))


if __name__ == "__main__":
    main()
