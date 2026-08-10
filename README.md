# AegisFlow 🛡️

**A Deterministic State Harness & Fault-Tolerant Orchestrator for Multi-Agent AI Workflows**

AegisFlow is a robust, production-grade orchestrator designed to solve the inherent unreliability of multi-agent LLM systems. By strictly separating state management from agent execution, AegisFlow guarantees that progress is never lost due to API timeouts, crashes, or hallucination loops.

## Engineering Highlights

*   **Durable State Patterns**: Designed a fault-tolerant multi-agent execution harness utilizing durable state patterns (via LangGraph and custom PostgreSQL checkpointers), guaranteeing zero loss of context or token progress during upstream API rate limits or node crashes.
*   **Deterministic Circuit Breakers**: Implemented strict schema enforcement and cyclic circuit breakers using Pydantic and state history logs to mitigate infinite agent recursion and hallucination feedback loops.
*   **Sandboxed Ephemeral Execution**: Built an isolated, ephemeral code-execution sandbox layer (using the Docker SDK) to safely test and evaluate multi-agent code generation outputs before merging.

## Architecture

1.  **State Management (Orchestration)**: Powered by LangGraph and PostgreSQL (`JSONB`), providing an immutable time-travel ledger of all agent states.
2.  **Execution (Agents)**: Independent, task-specific agents (Triage, Coder, Reviewer) powered by Qwen and strictly typed via Pydantic.
3.  **Sandbox**: Docker-based secure execution environment.

## Quickstart

### Prerequisites
*   Docker & Docker Compose
*   Python 3.10+
*   Qwen API Key (or OpenAI-compatible endpoint)

### Installation

```bash
# 1. Clone the repository
git clone <url>
cd harness_engineer

# 2. Start PostgreSQL
docker-compose up -d

# 3. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 4. Set Environment Variables
export GROQ_API_KEY="your-groq-api-key"
export AEGIS_MODEL="llama-3.1-8b-instant"
export DATABASE_URL="postgresql://aegis:password@localhost:5432/aegisflow"
```

### Running the API

```bash
uvicorn aegisflow.api.main:app --reload
```
