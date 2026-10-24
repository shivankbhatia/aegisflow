"""Durable checkpoint storage via the official LangGraph async Postgres saver."""
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@asynccontextmanager
async def postgres_checkpointer(dsn: str):
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()  # idempotent
        yield saver