# Router label review

I reread all 30 prompts under a frozen rubric: `direct` asks one answerable
fact or procedure from one documentation topic; `multi_hop` requires joining
distinct facts or procedures; `out_of_scope` asks for information or work the
indexed technical documentation does not cover. Each row in
`router_queries.jsonl` now includes a `label_rationale` and records this second
pass in `label_review`.

The re-review keeps the balanced split at 10 per route. It did not change any
labels. The query authorship remains Codex; the route labels received a
separate second-pass review by Codex. This does not claim independent human
annotation. The clean router test still has 19 cases because 11 results were
already observed; one exposed case is retained in the 20-row assignment but
excluded from scoring.
