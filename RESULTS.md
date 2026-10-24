# Phase 2 retrieval evaluation

## What these results establish

These are exploratory measurements on the current 81-question set, not an
independent benchmark. The set has 66 single-hop and 15 multi-hop questions.
Although all rows have `verified: true`, provenance records 72 as
"Codex (user-delegated)", 6 without a reviewer, and 3 as assistant-authored
questions. Most questions were generated from indexed chunks. Treat the set as
assistant-reviewed and source-derived, not as independently human-verified.
No held-out tuning set was used.

The corpus contains 946 documentation pages: 751 LangChain, 155 FastAPI, and
40 psycopg. Evaluation used the local pgvector index and made no generation API
calls. PyTorch MPS was unavailable in the offline run, so these retrieval
measurements used CPU inference.

## Dense chunker comparison at document level

Chunk IDs differ between chunking strategies, so this comparison maps each
retrieved chunk ID and each gold chunk ID to its document ID, preserving the
rank of the first result for each document. Each query over-fetches 50 chunks
before deduplication, so repeated hits do not artificially shorten the
document-level top-k. This avoids treating coincidental IDs such as `doc#3` as
equivalent chunks. It does not establish which chunk text is best for answer
generation.

| Dense strategy | Indexed chunks | Recall@1 | Recall@5 | Recall@10 | MRR | Retrieval p95 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-size | 7,389 | 0.673 | 0.938 | 0.957 | 0.850 | 56.3 ms |
| Structure-aware | 26,774 | 0.506 | 0.951 | **0.969** | 0.764 | 32.7 ms |
| Semantic | 14,242 | **0.747** | 0.951 | 0.951 | **0.887** | 32.2 ms |

No strategy dominates all measures. Structure-aware leads recall@10 while
semantic leads on MRR and recall@1. These measurements use the
existing source-derived question set, so they are not evidence of
generalization to independently authored questions.

## Exact served retrieval comparison at k=5

The API is configured for structure-aware chunks and caps its fused hybrid
candidate set to the request `k` before reranking. The table compares that
served behavior with dense-only retrieval at the same `k=5`, against the same
structure-aware gold chunk IDs.

| Configuration | Recall@5 | Hit@5 | MRR | Retrieval pipeline p95 |
|---|---:|---:|---:|---:|
| Dense structure-aware, k=5 | **0.852** | **0.889** | **0.668** | 37.9 ms |
| Hybrid, k=5 | 0.778 | 0.827 | 0.548 | 311.4 ms |
| Served hybrid + rerank, k=5 | 0.778 | 0.827 | 0.661 | 713.4 ms |

The served hybrid + rerank path is behind dense on recall@5 and hit@5, and its
MRR is slightly lower. It does not support a retrieval quality improvement
claim. Since the reranker sees only five fused candidates, it can reorder
those candidates but cannot increase recall beyond the candidate set. The
older 30-candidate offline score is not a measure of the served endpoint.

These are local CPU retrieval/reranking pipeline p95 values, not API time to
first token.

## Generation, judge, and live latency status

- No full answer-generation evaluation was run. Faithfulness ≥0.85 and citation
  validity ≥95% are therefore **not established**.
- `eval/judge_calibration.json` records 90% agreement on synthetic paired
  faithful/unfaithful examples, labeled by Codex under user delegation. It is
  not human judge calibration; one synthetic unfaithful answer received 1.0.
- A previous single live request measured 1,036.9 ms to first token and
  1,478.7 ms total, with 3 valid citations and no invalid markers. This is one
  sample, not a p95. A multi-request live measurement was skipped because of
  the Groq quota limit.
- An earlier 2,975 ms offline p95 came from a 30-candidate CPU reranking run.
  It is not product latency and is not representative of the served k=5 path.

## Reports

- Document-level fixed-size: [report](eval/results/doc-level-dense-fixed-size-20261024-125925.json)
- Document-level structure-aware: [report](eval/results/doc-level-dense-structure-aware-20261024-125927.json)
- Document-level semantic: [report](eval/results/doc-level-dense-semantic-20261024-125930.json)
- Served dense structure-aware k=5: [report](eval/results/served-baseline-dense-structure-aware-k5-20261024-125936.json)
- Served hybrid structure-aware k=5: [report](eval/results/served-hybrid-structure-aware-k5-20261024-125952.json)
- Served hybrid + rerank structure-aware k=5: [report](eval/results/served-hybrid-rerank-structure-aware-k5-20261024-130040.json)

## Resume bullet

Built a pgvector-backed RAG over 946 documentation pages with dense and hybrid
retrieval, RRF, a cross-encoder reranker, and a streaming API. In the current
81-question source-derived evaluation, served hybrid + rerank at k=5 scored
0.778 recall@5 versus 0.852 for dense-only; the chunker comparison was rerun at the
document level. Full answer faithfulness, citation validity, and live API p95
remain unmeasured.
