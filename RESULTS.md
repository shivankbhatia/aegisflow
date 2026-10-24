# Phase 2.6 retrieval ablation

## Evaluation set and corpus

- **81 verified questions**, including **15 multi-hop** questions.
- **946 documentation pages**: 751 LangChain, 155 FastAPI, and 40 psycopg.
- All retrieval runs used the same verified questions and `k = 1, 5, 10`.
- The local evaluation runtime had PyTorch MPS unavailable, so these offline
  retrieval measurements used CPU inference. They did not call Groq or generate
  answers.

## Dense chunker comparison

| Dense strategy | Indexed chunks | Recall@10 | MRR | Retrieval p95 |
|---|---:|---:|---:|---:|
| Fixed-size | 7,389 | 0.395 | 0.249 | 92.4 ms |
| Structure-aware | 26,774 | **0.889** | **0.673** | 34.6 ms |
| Semantic | 14,242 | 0.352 | 0.199 | 33.8 ms |

Structure-aware is the best dense chunker by both recall@10 and MRR.

## Dense → hybrid → reranker comparison

The following comparison uses structure-aware chunks, the best dense strategy.
Hybrid retrieval uses dense+sparse RRF; reranking uses 30 candidates.

| Metric | Dense (structure-aware) | Hybrid | Hybrid + rerank |
|---|---:|---:|---:|
| Recall@1 | 0.432 | 0.309 | 0.463 |
| Recall@5 | 0.852 | 0.778 | 0.901 |
| Recall@10 | 0.889 | 0.889 | **0.944** |
| Hit@1 | 0.494 | 0.370 | 0.531 |
| Hit@5 | 0.889 | 0.827 | 0.938 |
| Hit@10 | 0.926 | 0.926 | 0.975 |
| MRR | 0.673 | 0.561 | **0.707** |
| Retrieval p95 | 34.6 ms | 316.2 ms | 2,974.7 ms |

Compared with dense structure-aware retrieval, hybrid plus reranking raised
recall@10 by **0.055** and MRR by **0.034**. Hybrid alone preserved recall@10
but lowered MRR by 0.112; the cross-encoder recovered and exceeded the dense
baseline. The offline 30-candidate reranker p95 is high on this CPU runtime.

By question type, hybrid plus reranking reached recall@10 / MRR of 0.970 / 0.689
for single-hop and 0.833 / 0.785 for multi-hop questions.

## Live API latency sample

The optimized `/query` endpoint was measured once with `k=5` on MPS:

| Retrieval | Rerank | Time to first token | Total response |
|---:|---:|---:|---:|
| 106.9 ms | 237.7 ms | 1,036.9 ms | 1,478.7 ms |

The sample returned three cited chunks and no invalid citation markers. This is
one request, **not a p95 measurement**. A multi-request p95 run was skipped to
conserve Groq API quota. The full generation/evaluation faithfulness checks are
also separate and are not represented by these retrieval metrics.

## Saved reports

- [Dense fixed-size](eval/results/dense-fixed-size-20261024-123110.json)
- [Dense structure-aware](eval/results/dense-structure-aware-20261024-123112.json)
- [Dense semantic](eval/results/dense-semantic-20261024-123114.json)
- [Hybrid structure-aware](eval/results/hybrid-structure-aware-20261024-123130.json)
- [Hybrid plus reranker, structure-aware, 30 candidates](eval/results/hybrid-reranked-structure-aware-30-20261024-123506.json)

## Resume bullet

Built a hybrid-retrieval RAG over **946 documentation pages** with pgvector,
RRF fusion, and a cross-encoder reranker; raised recall@10 from **0.889 to
0.944** and MRR from **0.673 to 0.707** on an **81-question verified golden
set**, with **2,975 ms offline retrieval p95** on CPU. A separate single live
API sample measured 1,037 ms to first token; live API p95 was not measured.
