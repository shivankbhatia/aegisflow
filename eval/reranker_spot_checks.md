# Reranker spot checks

Candidate count: 30; reranked p95: 2316.0 ms; added p95: 2246.1 ms.

| candidates | MRR | recall@10 | total p95 | added p95 |
|---:|---:|---:|---:|---:|
| 50 | 0.810 | 0.938 | 3063.5 ms | 2993.5 ms |
| 30 | 0.819 | 0.944 | 2316.0 ms | 2246.1 ms |

Reducing to 30 candidates improved MRR by 0.009 and recall@10 by 0.006,
while reducing added p95 by 747.4 ms. The 30-candidate latency still exceeds
the 300 ms target on this machine.

## s00_000: What is an environment variable and why is it commonly used for external settings in Python applications?
- Gold: fastapi/advanced/settings#0
- Hybrid top 5: fastapi/advanced/settings#0, fastapi/advanced/settings#6, fastapi/advanced/settings#2, fastapi/advanced/settings#9, fastapi/environment-variables#0
- Reranked top 5: fastapi/environment-variables#0, fastapi/advanced/settings#0, fastapi/advanced/settings#1, fastapi/advanced/settings#9, fastapi/advanced/settings#6
- Assessment: Right: the reranker moved the concise Environment Variables page to rank 1 while keeping the cited definition chunk at rank 2.

## s00_001: What does the log output show when Uvicorn starts up with multiple workers?
- Gold: fastapi/deployment/server-workers#4
- Hybrid top 5: fastapi/deployment/server-workers#3, fastapi/deployment/server-workers#5, fastapi/deployment/server-workers#1, fastapi/deployment/server-workers#0, fastapi/deployment/server-workers#6
- Reranked top 5: fastapi/deployment/server-workers#6, fastapi/deployment/server-workers#1, fastapi/deployment/server-workers#5, fastapi/deployment/server-workers#4, fastapi/tutorial/first-steps#3
- Assessment: Right: it pulled the exact log-output chunk from outside the hybrid top 5 to rank 4, where it directly answers the question.

## s00_003: What happens if you try to access the FastAPI docs UI at http://127.0.0.1:8000/docs instead of the proxy URL http://127.0.0.1:9999/api/v1/docs?
- Gold: fastapi/advanced/behind-a-proxy#12
- Hybrid top 5: fastapi/advanced/behind-a-proxy#12, fastapi/advanced/behind-a-proxy#13, fastapi/advanced/behind-a-proxy#5, fastapi/advanced/behind-a-proxy#4, fastapi/tutorial/metadata#0
- Reranked top 5: fastapi/advanced/behind-a-proxy#12, fastapi/advanced/sub-applications#4, fastapi/advanced/behind-a-proxy#13, fastapi/advanced/behind-a-proxy#4, fastapi/index#13
- Assessment: Right: the cited proxy-specific chunk stayed at rank 1; the reranker only changed the distractors below it.

## s00_005: Which key security feature should be prioritized when deploying a FastAPI application to ensure safe communication with API clients?
- Gold: fastapi/deployment/concepts#0
- Hybrid top 5: fastapi/deployment/concepts#0, fastapi/tutorial/security/index#5, fastapi/reference/security/index#0, fastapi/tutorial/security/first-steps#0, fastapi/tutorial/security/oauth2-jwt#13
- Reranked top 5: fastapi/features#5, fastapi/tutorial/security/first-steps#5, fastapi/deployment/concepts#0, fastapi/tutorial/security/oauth2-jwt#13, fastapi/tutorial/security/index#5
- Assessment: Wrong: a generic authentication-features chunk outranked the deployment passage that explicitly lists HTTPS, dropping the gold from rank 1 to rank 3.

## s00_006: Which FastAPI version range supports using both Pydantic v1 and v2 models in the same application during migration?
- Gold: fastapi/how-to/migrate-from-pydantic-v1-to-pydantic-v2#8
- Hybrid top 5: fastapi/how-to/migrate-from-pydantic-v1-to-pydantic-v2#4, fastapi/how-to/migrate-from-pydantic-v1-to-pydantic-v2#0, fastapi/how-to/migrate-from-pydantic-v1-to-pydantic-v2#8, fastapi/how-to/migrate-from-pydantic-v1-to-pydantic-v2#5, fastapi/how-to/migrate-from-pydantic-v1-to-pydantic-v2#7
- Reranked top 5: fastapi/how-to/migrate-from-pydantic-v1-to-pydantic-v2#8, fastapi/how-to/migrate-from-pydantic-v1-to-pydantic-v2#0, fastapi/how-to/migrate-from-pydantic-v1-to-pydantic-v2#5, fastapi/how-to/migrate-from-pydantic-v1-to-pydantic-v2#4, fastapi/how-to/migrate-from-pydantic-v1-to-pydantic-v2#7
- Assessment: Right: it moved the exact Pydantic migration version-range chunk from rank 3 to rank 1.
