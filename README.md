# AegisFlow

A fault-tolerant orchestrator for multi-agent LLM workflows. Agents run as nodes in a LangGraph state machine whose state is checkpointed to PostgreSQL after every step, so a crashed or restarted run resumes from its last checkpoint without re-executing completed nodes.

## What it does

A Triage → Coder → Reviewer pipeline that turns a task description into code plus tests, executes both in a network-isolated Docker sandbox, and loops on reviewer feedback until the patch is approved or a guardrail trips.

- **Durable state:** LangGraph + the official async Postgres checkpointer. Crash recovery is tested: kill the reviewer mid-run, resume, and triage and coder do not re-run (`tests/test_crash_recovery.py`, `tests/test_api.py`).
- **Guardrails:** a retry circuit breaker (counts coder attempts, default 3), a per-run token budget checkpointed in graph state (default 50,000), and sandbox-result caching for identical code.
- **LLM gateway** (`llm_gateway/`): provider-agnostic async client over raw `httpx` for any OpenAI-compatible endpoint. Exponential backoff with jitter and `Retry-After` support, bounded concurrency, SSE streaming, token counting, per-call usage logging.
- **Structured outputs:** Pydantic schemas enforced via `json_schema` response format, with a validate → repair → retry loop and a prompt-mode fallback.
- **Sandbox:** generated code runs in an ephemeral Docker container with no network and a 128 MB memory limit.
- **API:** FastAPI service to start, inspect, resume and stream runs.

Measured on Groq with `openai/gpt-oss-20b`: 20 concurrent calls complete in about 2.7 s (roughly 13 s serial); the provider adds a fixed 67-token chat-template overhead per request, which the gateway's token estimate accounts for.

## Architecture

```
POST /run ──► RunManager ──► LangGraph ──► triage ──► coder ──► reviewer ──┐
                  │             │                        ▲                 │ rejected
   WS /ws/events  │             │ checkpoint             └─────────────────┘ (max 3 attempts)
                  ▼             ▼ after each node
             live events    PostgreSQL ◄── GET /runs/{id}, POST /runs/{id}/resume
                                    
   agents ──► llm_gateway (retries, backoff, structured outputs) ──► Groq / any OpenAI-compatible API
   reviewer ──► Docker sandbox (no network)
```

## Quickstart

Requires Python 3.11+, Docker, and a Groq API key (or any OpenAI-compatible endpoint).

```bash
git clone https://github.com/shivankbhatia/aegisflow && cd aegisflow
docker-compose up -d                      # PostgreSQL on :5432
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp /dev/null .env                         # then add the variables below
uvicorn aegisflow.api.main:app --reload
```

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (required) | API key for the endpoint |
| `AEGIS_MODEL` | `openai/gpt-oss-20b` | Model name |
| `AEGIS_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible base URL |
| `AEGIS_PROMPT_OVERHEAD` | measured per model | Fixed prompt tokens added by the provider |
| `DATABASE_URL` | `postgresql://aegis:password@localhost:5432/aegisflow` | Checkpoint database |
| `AEGIS_RAG_ENABLED` | `0` | Load the local BGE embedder and cross-encoder once during FastAPI startup |

## API

| Endpoint | Description |
|---|---|
| `POST /run` `{"task_description": "..."}` | Start a run; returns `run_id` (202) |
| `GET /runs/{id}` | Status (`queued`, `running`, `completed`, `failed`, `interrupted`), plan, patches, attempts, tokens |
| `POST /runs/{id}/resume` | Resume a `failed` or `interrupted` run from its last checkpoint |
| `WS /ws/events/{id}` | Per-node events for the current process, then `done` or `error` |
| `GET /health` | Liveness |

```bash
curl -XPOST localhost:8000/run -H 'content-type: application/json' \
  -d '{"task_description": "Fix hello() so it returns Hello, World!"}'
```

Run state is durable in Postgres. Live events and error text are in memory, so after a server restart a run shows as `interrupted` and can be resumed, but its past events are not replayed.

## Tests

```bash
pytest -q                                # 35 tests; Postgres-backed tests skip if it is down
pytest tests/test_e2e_live.py            # live end-to-end: needs GROQ_API_KEY and Docker
python scripts/gateway_smoke.py          # live gateway: concurrency, token calibration, streaming
python scripts/structured_smoke.py       # live structured extraction: validity and accuracy
```

## Layout

```
src/aegisflow/
  agents/        triage, coder, reviewer (async, structured outputs)
  llm_gateway/   async client, retries, streaming, structured outputs
  orchestrator/  LangGraph workflow, guardrails, Postgres checkpointer
  sandbox/       Docker executor
  api/           FastAPI app, run manager, routes
```

## Roadmap

Retrieval-augmented generation over a document corpus (hybrid search, reranking, evaluation harness), exposed to the agents as a tool.

### Hybrid retrieval ranking note

The RAG index combines pgvector cosine search with PostgreSQL full-text search
using the `simple` configuration, then fuses their rankings with Reciprocal Rank
Fusion (RRF). PostgreSQL's `ts_rank_cd` is **not true BM25**. If BM25 is worth
testing later, compare `pg_search` or an in-memory `rank_bm25` index as an
experiment.

### Cross-encoder reranking

Hybrid retrieval can rerank its top 50 candidates with
`BAAI/bge-reranker-base`. The cross-encoder truncates each query/passage pair
to 512 tokens, scores batches of 16, and uses MPS when PyTorch reports it
available (otherwise CPU). Set `AEGIS_RAG_ENABLED=1` to load the embedder and
reranker once in the FastAPI lifespan. Run `python scripts/evaluate_reranker.py`
to compare MRR, recall@10, and latency with the hybrid baseline; it retries
with 30 candidates if 50 adds at least 300 ms at p95.
