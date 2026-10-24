# Phase 2 retrieval evaluation

## Phase A selection rule (written before the next evaluation run)

Among the nine served-mode/chunker combinations, select the configuration with
the highest document-level recall@5 whose measured retrieval p95 is below
150 ms. Compare that candidate with the incumbent dense-only,
structure-aware configuration using a paired bootstrap over question-level
recall@5 differences. If the 95% interval includes zero, keep the incumbent
dense-only configuration; it is simpler than adding hybrid retrieval or a
reranker. This rule will also govern the 15- and 30-candidate reranker tradeoff
runs. The rule selects **dense / structure-aware**. It had the highest
recall@5 among configurations below the latency cutoff (0.938; p95 32.9 ms).
Every other latency-feasible configuration has a paired 95% interval versus
this incumbent that includes zero, so dense-only remains selected. This is a
dev-set decision, not a held-out result.

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

## Phase A live citation spot-check

Five live `/query` calls ran on the selected dense / structure-aware service.
Four returned topical page titles. One LangChain callback question returned
`trace-with-langchain` and an invalid `[0]` marker. The LangGraph `persistence`
and Psycopg `pool` pages had empty heading paths. TTFT p50 was 836 ms and p95
was 5,654 ms for these five samples; this small spot-check is not a stable
latency benchmark. Details are in the [sample report](eval/results/phase-a-live-citation-check.json).

## Served document-level comparison at k=5

All nine configurations use the same k=5 output, with document IDs scored
against document-level gold. Dense and hybrid retrieve five chunks; hybrid +
rerank uses a five-chunk candidate pool and returns five chunks.

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

The 15-candidate and 30-candidate reranker runs on structure-aware chunks
scored 0.957 and 0.944 document recall@5, with p95 1,607.5 ms and 2,928.9 ms.
Both miss the 150 ms latency limit. The reranker can only reorder its
candidate pool.

These are local CPU retrieval/reranking pipeline p95 values, not API time to
first token. The paired bootstrap matched questions by ID over 10,000
resamples (seed 20261024). Differences are candidate minus dense/structure-aware:

| Candidate | Recall@5 difference | 95% CI | Better / worse questions |
|---|---:|---:|---:|
| Dense / fixed-size | -0.012 | [-0.062, 0.037] | 3 / 4 |
| Dense / semantic | -0.019 | [-0.068, 0.025] | 2 / 4 |
| Hybrid / fixed-size | -0.019 | [-0.074, 0.031] | 4 / 5 |
| Hybrid / structure-aware | -0.025 | [-0.068, 0.006] | 1 / 3 |
| Hybrid / semantic | -0.019 | [-0.074, 0.031] | 3 / 4 |
| Hybrid + rerank / structure-aware | -0.025 | [-0.068, 0.006] | 1 / 3 |

All listed intervals include zero. Counts omit tied questions.

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
- Bootstrap summary: [report](eval/results/served-doc-bootstrap-20261024.json)
- Dense fixed-size: [report](eval/results/served-doc-dense-fixed-size-k5-20261024-165839.json)
- Hybrid fixed-size: [report](eval/results/served-doc-hybrid-fixed-size-k5-20261024-165846.json)
- Hybrid + rerank fixed-size: [report](eval/results/served-doc-hybrid_rerank-fixed-size-k5-20261024-165929.json)
- Dense structure-aware: [report](eval/results/served-doc-dense-structure-aware-k5-20261024-165932.json)
- Hybrid structure-aware: [report](eval/results/served-doc-hybrid-structure-aware-k5-20261024-165947.json)
- Hybrid + rerank structure-aware: [report](eval/results/served-doc-hybrid_rerank-structure-aware-k5-20261024-170035.json)
- Dense semantic: [report](eval/results/served-doc-dense-semantic-k5-20261024-170038.json)
- Hybrid semantic: [report](eval/results/served-doc-hybrid-semantic-k5-20261024-170047.json)
- Hybrid + rerank semantic: [report](eval/results/served-doc-hybrid_rerank-semantic-k5-20261024-170130.json)
- Reranker pool 15: [report](eval/results/served-doc-hybrid-rerank-structure-aware-k5-pool15-20261024-170332.json)
- Reranker pool 30: [report](eval/results/served-doc-hybrid-rerank-structure-aware-k5-pool30-20261024-170703.json)

## Resume bullet

Built a pgvector-backed RAG over 946 documentation pages with dense and hybrid
retrieval, RRF, a cross-encoder reranker, and a streaming API. In the current
81-question source-derived evaluation, dense structure-aware was selected at
0.938 document recall@5 and 32.9 ms retrieval p95; hybrid and reranking failed
the paired quality or latency rule. Five live requests had a 5,654 ms TTFT p95
and exposed one retrieval/citation issue. Held-out answer evaluation and
larger-sample live API p95 remain unmeasured.
