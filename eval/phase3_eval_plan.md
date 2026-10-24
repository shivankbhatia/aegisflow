# Phase 3 QA evaluation setup

## Frozen baseline set

Use `eval/golden.jsonl` as the primary development set: 81 questions total, 66
single-hop and 15 multi-hop. The set includes reference answers and gold chunk
IDs; retrieval coverage is scored at document level by comparing each gold
document ID with the retrieved documents. The existing 15 items in
`eval/handwritten_questions.jsonl` are an optional supplement, but currently
have neither reference answers nor gold documents. Keep them out of aggregate
quality numbers until those fields are supplied.

This is a development baseline, not a held-out test set. Existing `verified`
flags do not establish independent authorship or human validation.

## Retrieval-only baseline (quota-free)

Run `python scripts/evaluate_retrieval_coverage.py`. The saved report at
`eval/results/phase3-retrieval-baseline.json` covers all 81 questions with the
served retriever at k=10 and records the model/mode/chunking setup. Its 15-case
multi-hop all-gold-document rate is 0.80 (12/15). This is separate from answer
generation and makes no LLM calls.

`python scripts/evaluate_qa_multihop.py` is the later planner-only comparison:
it plans and retrieves for the 15 multi-hop questions, performs no synthesis,
and uses a paired bootstrap against the retrieval-only report.

## Generation baseline

Run `python scripts/phase3_baseline.py` against a running API and its configured
PostgreSQL index. It calls the existing `/query` endpoint at `k=10`, measures
retrieval coverage, answer correctness, faithfulness, token count, and end-to-end
latency. Judge-scored metrics must be reported as **LLM-judged, uncalibrated**
until human labels are completed. The report at
`eval/results/phase3-baseline.json` is resumable; `complete: true` is required
before treating its aggregates as the full baseline.

The intended generation comparison is the explicit 35-question file
`eval/phase3_gen35.jsonl` (15 multi-hop and 20 single-hop), using
`AEGIS_GOLDEN_SET=eval/phase3_gen35.jsonl`,
`AEGIS_EVAL_EXPECTED_COUNT=35`, and
`AEGIS_PHASE3_BASELINE_OUTPUT=phase3-generation-baseline-35.json`.
The prior incomplete baseline contains one generated answer at 2,751 tokens;
that is only a rough per-answer estimate (96,285 generation tokens for 35),
and it has no saved judge-token usage. Do not treat this estimate as the full
quota estimate. Router, planner, and judge calls must also be included, using
their saved per-call token means, before any generation run. Compare the total
with remaining tokens for the exact model recorded in the report.

## Router test validity and model quota

`eval/results/qa-router-v1-partial.json` records 11 v1 results (10/11 correct)
as a partial run, not a clean test. The v1 prompt snapshot was not saved, and
the prompt changed after those queries were observed. `eval/router_test20.jsonl`
contains 20 rows: 19 untouched cases and one exposed multi-hop case retained
with `test_eligible: false`. The v2 evaluator scores only the 19 untouched
cases and records the frozen prompt hash and model. This is the largest clean
held-out set possible from the 30 rows after 11 were exposed; it is reported as
19 clean cases, not 20. The questions and intended-scope labels were
Codex-authored, not independently human-labeled.

Groq quota is model-specific for planning purposes. Keep the model name in
every report and budget each stage against that model's observed remaining
daily quota. Do not count the retrieval-only baseline as quota-consuming.

The script writes `eval/results/phase3-human-review.jsonl` after the complete
run. Label 15 answers for correctness and faithfulness, then calculate judge
agreement separately for both metrics. Do not use the existing synthetic,
Codex-labeled `eval/judge_calibration.json` as human calibration.

## Pre-registered decision rules

Record these rules before collecting calibration or classifier results; do not
change them after inspecting holdout outcomes.

- **Verifier threshold:** use only the 15 human-labeled calibration answers.
  The verifier passes an answer when `score >= threshold` and citations are
  valid. Choose the lowest threshold in `[0, 1]` for which zero human-labeled
  unfaithful answers pass. If no threshold satisfies that rule, report the
  judge as unsuitable for threshold selection instead of relaxing the rule.
  Also report the false-refusal rate on human-labeled faithful answers at the
  selected threshold and citation-invalid false refusals separately.
- **Revision limit:** freeze at two revisions (at most three synthesis drafts)
  before calibration; after the second unsuccessful verification, refuse with
  `insufficient evidence`. Do not tune this limit on evaluation results.
- **Golden-question false flags:** a false flag is any classifier outcome that
  blocks a question in the answerable 66-item golden single-hop set. Invalid,
  truncated, or unavailable classifier output counts as a block as well as a
  positive verdict; report those causes separately. Sort IDs by
  `SHA256("classifier-split-v1:" + id)`: first 33 are the tuning half and the
  remaining 33 are the untouched report half. Any token-cap adjustment is made
  using only the first half, then frozen before the second half is run.

Run the classifier's 33-case tuning split first with
`python scripts/evaluate_injection_classifier.py --split tuning`. Inspect both
positive flags and `finish_reason`; `length`, invalid JSON, or provider errors
count as false flags for benign questions. Adjust only `classifier_max_tokens`
on these 33 cases if truncation occurs. Then freeze that cap and run
`--split holdout`; only this untouched half supplies the reported golden false
flag rate. The evaluator records finish reasons, token use, invalid outputs,
and checkpoints after every case; it stops after three consecutive 429s and
resumes by retrying those failed IDs.

The blind router relabel uses `eval/router_blind_prompts.jsonl` with shuffled
neutral IDs and labels in `eval/router_blind_labels.jsonl`. Its mapping is kept
separately in `eval/router_blind_id_map.jsonl`; the labeler did not see the map
while assigning labels. It was done in the same session rather than after the
suggested 24-hour washout, and the assistant had previously seen these
questions. Treat the result as a neutral-ID second pass, not a fully independent
human relabel. It agrees with the Codex labels on 30/30, which is not evidence
of independent accuracy. The comparison report is
`eval/results/router-blind-relabel.json`.

## QA checkpoint state

`src/aegisflow/rag/qa_state.py` defines the compact QA state contract. Evidence
keeps the chunk ID, source, heading path, and trimmed text only; full chunk
records should stay in the retrieval store and out of Postgres checkpoints.

## Deterministic guardrail fixtures

The 30 direct injection prompts, 10 poisoned chunks, and 10 unanswerable
questions are in `eval/security/`. Deterministic coverage tests validate the
input injection heuristics, citation markers, and evidence delimiting/escaping
without claiming LLM classifier or answer-quality performance. All judge-based
metrics remain **LLM-judged, uncalibrated** until 15 human labels are completed.
`python scripts/evaluate_qa_deterministic.py` saves these deterministic counts
to `eval/results/qa-deterministic-guardrails.json`; it explicitly leaves the
unanswerable refusal rate unmeasured until graph/model eval is run.

The original 30 injection prompts were used during heuristic development and
are not an independent test. Their 30/30 block result is a tuned-set result.
The 15 prompts in `eval/security/direct_injection_seen15.jsonl` were written
after the rules were frozen, but before the classifier prompt was frozen. They
are now marked as a seen heuristic check: 0/15 were caught by the rules. A new
15-item challenge set must be written only after the combined rules and
classifier prompt are frozen in 3-D. The saved deterministic report includes a
run manifest; its 0/66 false-flag result on the golden single-hop set is kept.

The QA graph now runs the frozen regex layer before a structured injection
classifier. Positive, invalid, or unavailable classifier output fails closed;
the regex layer short-circuits without a model call. Fakes cover positive,
negative, invalid, and short-circuit behavior. See
[`docs/threat_model.md`](../docs/threat_model.md) for assets, mitigations, and
residual risks. Real classifier and poisoned-evidence model evaluations remain
quota-bound and are not claimed by deterministic tests.

The regex rules and classifier prompt are frozen together in
`eval/security/guardrail_freeze_v1.json`; tests verify both hashes. Only after
that freeze, the new 15-case challenge was authored in
`eval/security/direct_injection_challenge15_v1.jsonl`. The deterministic report
shows 0/15 regex blocks; classifier performance has not been measured. The
prior 15-case set remains `direct_injection_seen15.jsonl`. Structured
classifier results preserve `finish_reason`, including `length` on truncation,
so the first quota run can detect legitimate-question refusals caused by a
too-small output cap.

## Phase 3-B test hygiene

`pytest -q` excludes `live` and `heavy` tests by default. The full default suite
was run with `GROQ_API_KEY` unset and PostgreSQL/Docker available:
119 passed, with only the explicitly marked live test deselected. Run live tests
with `pytest -m live`; local model loads use `pytest -m heavy` after installing
the optional `.[models]` extra. GitHub Actions configures Python 3.11 and the
PostgreSQL 15 pgvector service and runs the default tests without secrets; a
hosted push result is still needed to claim CI green.

The new manifest helper records git SHA/dirty state, model, endpoint host, UTC
date, temperature, Python, QA config version, and key library versions.
`save_report` rejects a missing or incomplete manifest. The saved older reports
predate this protocol; newly generated reports include the manifest.

The tunable QA prompts and limits now live in `src/aegisflow/rag/qa_config.py`
and are passed through the graph. `qa-v0.1-provisional` is not frozen: its
verifier threshold and revise limit must still be calibrated using the planned
30 answers and 15 independent human labels. Do not tag or use this version as
the final frozen config yet.

The file `eval/router_relabels_second_pass.jsonl` records assistant-written
route labels and rationales; `scripts/compare_router_labels.py` prints any
disagreement with the original Codex labels. This is not a blind relabel: the
original IDs encode each route, so the labels are not independent and cannot
replace a human label pass. The 19-clean-case router run still awaits quota.
The 15 independent multi-hop questions are in
`eval/multihop_independent15.jsonl`; their source document IDs were checked
against the local corpus, but they are assistant-authored, not human-reviewed.
The primary 12/15 misses and the independent set's 13/15 misses are analyzed in
`eval/retrieval_failure_review.md`. The independent retrieval-only baseline is
saved at `eval/results/multihop-independent15-retrieval-baseline.json`.
