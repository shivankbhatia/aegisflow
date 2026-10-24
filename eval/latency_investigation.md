# Live query latency investigation

## Diagnosis

The query initially took about 7–8 seconds to return its first token. Stage
timings showed that reranking on CPU was the main bottleneck: reranking 50
candidates took 6,356 ms. Retrieval took 91 ms; first token arrived at 7,238 ms.

The failed requests in the server log were a separate issue. Groq returned HTTP
429 because the request needed 2,387 tokens while only 2,180 tokens remained
under the configured 200,000-token daily limit. Since `/query` streams its
response, Uvicorn logged HTTP 200 after the stream began; the failure was sent
as an SSE `error` event.

## Changes

- Serialize embedding and reranker inference with a shared lock when they use
  MPS, and allow `AEGIS_RAG_DEVICE` to select `mps` or `cpu`.
- Default the maximum reranker candidate pool to 30, configurable with
  `AEGIS_RAG_CANDIDATE_K` (1–50).
- For live `/query` requests, limit reranking to the requested result count
  (`k=5` by default). Evaluation retrievers keep the configured candidate pool.
- Include retrieval, rerank, first-token, and total timings in successful
  `done` SSE events; include elapsed stage timings in generation `error` events.

## Measurements

| Configuration | Retrieval | Rerank | First token | Total |
|---|---:|---:|---:|---:|
| CPU, 50 candidates | 91 ms | 6,356 ms | 7,238 ms | 7,794 ms |
| MPS, 30 candidates | 372 ms | 2,287 ms | 3,283 ms | 3,768 ms |
| MPS, live `/query` `k=5` candidate cap | 107 ms | 238 ms | 1,037 ms | 1,479 ms |

The final sample met the 1.5-second first-token target and returned three
citation references with no invalid markers. These are individual samples, not
a p95 result.

## Remaining check

The 20-request p95 measurement was skipped to conserve Groq API quota. The
reported `tokens` value for the final sample was 1,219, so a 20-sample run could
consume roughly 24,000 tokens. Do not treat the single 1,037 ms observation as
a p95 measurement.
