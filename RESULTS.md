# RAG evaluation results

## Dev set and held-out set

The available evaluation set contains 81 questions (66 single-hop and 15
multi-hop). Tuning and evaluation used the same 81 questions. The separate
held-out set has not yet been created and reviewed, so it has not been run.
The 81-question results below are exploratory dev-set measurements, not an
independent estimate of generalization.

The corpus used for these runs contains 946 documentation pages: 751
LangChain, 155 FastAPI, and 40 psycopg. Evaluation used the local pgvector
index and made no generation API calls. PyTorch MPS was unavailable, so the
retrieval/reranking runs used CPU inference.

## Decision rule and outcome

Before the comparison, the rule was set to choose the configuration with the
highest document-level recall@5 among configurations with retrieval p95 below
150 ms. The candidate is compared with dense-only / structure-aware using a
paired bootstrap over question-level recall@5. If the 95% interval includes
zero, choose the simpler dense-only configuration.

The rule selected **dense / structure-aware**: document recall@5 was 0.938 at
32.9 ms p95 in the served-mode comparison. Every latency-feasible alternative
had a paired 95% interval against this incumbent that included zero. Thus the
hybrid and reranking additions were not selected. This is a dev-set decision;
there is no held-out result yet.

## Served configuration comparison (k=5)

Every mode returns five chunks. Hybrid + rerank uses a five-chunk candidate
pool, matching the served path. Scores map retrieved and gold chunk IDs to
document IDs and preserve each document's first rank.

| Mode | Chunker | Document recall@5 | Retrieval p95 |
|---|---|---:|---:|
| Dense | Fixed-size | 0.926 | 38.4 ms |
| Hybrid | Fixed-size | 0.920 | 119.4 ms |
| Hybrid + rerank | Fixed-size | 0.920 | 606.0 ms |
| Dense | Structure-aware | **0.938** | 32.9 ms |
| Hybrid | Structure-aware | 0.914 | 319.8 ms |
| Hybrid + rerank | Structure-aware | 0.914 | 705.4 ms |
| Dense | Semantic | 0.920 | 33.8 ms |
| Hybrid | Semantic | 0.920 | 230.9 ms |
| Hybrid + rerank | Semantic | 0.920 | 666.3 ms |

The 15- and 30-candidate structure-aware reranker runs achieved document
recall@5 of 0.957 and 0.944, with p95 of 1,607.5 ms and 2,928.9 ms. Both
exceeded the latency limit. The reranker can only reorder its candidate pool.

The paired bootstrap matched question IDs over 10,000 resamples (seed
20261024). Differences are candidate minus dense / structure-aware; tied
questions are omitted from better/worse counts.

| Candidate | Recall@5 difference | 95% CI | Better / worse |
|---|---:|---:|---:|
| Dense / fixed-size | -0.012 | [-0.062, 0.037] | 3 / 4 |
| Dense / semantic | -0.019 | [-0.068, 0.025] | 2 / 4 |
| Hybrid / fixed-size | -0.019 | [-0.074, 0.031] | 4 / 5 |
| Hybrid / structure-aware | -0.025 | [-0.068, 0.006] | 1 / 3 |
| Hybrid / semantic | -0.019 | [-0.074, 0.031] | 3 / 4 |
| Hybrid + rerank / structure-aware | -0.025 | [-0.068, 0.006] | 1 / 3 |

All listed intervals include zero.

## Dense chunker comparison at document level

Chunk IDs differ by chunking strategy. This comparison maps each result and
gold chunk to its document ID, preserving the rank of the first result for
each document. Each query fetches 50 chunks before deduplication. It measures
document retrieval, not which chunk text is best for answer generation.

| Dense strategy | Indexed chunks | Recall@1 | Recall@5 | Recall@10 | MRR | Retrieval p95 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-size | 7,389 | 0.673 | 0.938 | 0.957 | 0.850 | 56.3 ms |
| Structure-aware | 26,774 | 0.506 | 0.951 | **0.969** | 0.764 | 32.7 ms |
| Semantic | 14,242 | **0.747** | 0.951 | 0.951 | **0.887** | 32.2 ms |

No strategy leads on every metric: structure-aware has the highest recall@10,
while semantic has the highest MRR and recall@1. This comparison uses the
same dev set used for tuning.

## Answer evaluation and live checks

- No full answer-generation evaluation was run. Faithfulness ≥0.85 and
  citation validity ≥95% have not been measured.
- The existing judge calibration file reports 90% agreement on synthetic
  faithful/unfaithful pairs labeled by Codex under user delegation. It is not
  human calibration; one synthetic unfaithful answer received a score of 1.0.
- Five live `/query` calls were spot-checked. Four returned topical page
  titles; one LangChain callback question returned an off-target page and an
  invalid `[0]` marker. The LangGraph `persistence` and Psycopg `pool` pages
  had empty heading paths.
- In those five samples, TTFT p50 was 836 ms and p95 was 5,654 ms. This sample
  is too small to treat as a stable latency benchmark. A previous single
  request measured 1,036.9 ms TTFT and 1,478.7 ms total with three valid
  citations. A larger live latency run was skipped because of the Groq quota
  limit.

## Limitations

The development set is source-derived and not independently human-verified.
Results cannot establish held-out retrieval quality or answer faithfulness.
CPU measurements are offline retrieval/reranking timings, not product TTFT.
See [README Limitations](README.md#limitations) for provenance and evaluation
caveats.

## Linked report index

All reports linked here are in `eval/results/`. Older or superseded reports
are retained in `eval/results/archive/`.

- **Dev chunker comparison:** [fixed-size](eval/results/doc-level-dense-fixed-size-20261024-125925.json), [structure-aware](eval/results/doc-level-dense-structure-aware-20261024-125927.json), and [semantic](eval/results/doc-level-dense-semantic-20261024-125930.json).
- **Served k=5 comparison:** [bootstrap](eval/results/served-doc-bootstrap-20261024.json); [dense fixed-size](eval/results/served-doc-dense-fixed-size-k5-20261024-165839.json), [hybrid fixed-size](eval/results/served-doc-hybrid-fixed-size-k5-20261024-165846.json), [hybrid + rerank fixed-size](eval/results/served-doc-hybrid_rerank-fixed-size-k5-20261024-165929.json); [dense structure-aware](eval/results/served-doc-dense-structure-aware-k5-20261024-165932.json), [hybrid structure-aware](eval/results/served-doc-hybrid-structure-aware-k5-20261024-165947.json), [hybrid + rerank structure-aware](eval/results/served-doc-hybrid_rerank-structure-aware-k5-20261024-170035.json); [dense semantic](eval/results/served-doc-dense-semantic-k5-20261024-170038.json), [hybrid semantic](eval/results/served-doc-hybrid-semantic-k5-20261024-170047.json), and [hybrid + rerank semantic](eval/results/served-doc-hybrid_rerank-semantic-k5-20261024-170130.json).
- **Reranker candidate tradeoff:** [pool 15](eval/results/served-doc-hybrid-rerank-structure-aware-k5-pool15-20261024-170332.json) and [pool 30](eval/results/served-doc-hybrid-rerank-structure-aware-k5-pool30-20261024-170703.json).
- **Live citation spot-check:** [sample report](eval/results/phase-a-live-citation-check.json).

## Resume bullet

Not written yet: the template requires dev and held-out numbers, and the
held-out set has not been run. Add the bullet after a reviewed held-out run.
