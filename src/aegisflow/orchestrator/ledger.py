"""Durable checkpoint storage.

Uses the official LangGraph Postgres saver instead of a hand-rolled
BaseCheckpointSaver: it implements the full interface (put_writes, blobs,
list, pending writes) that crash-resume depends on.
"""
from contextlib import contextmanager
from typing import Generator

from langgraph.checkpoint.postgres import PostgresSaver


@contextmanager
def postgres_checkpointer(dsn: str) -> Generator[PostgresSaver, None, None]:
    with PostgresSaver.from_conn_string(dsn) as saver:
        saver.setup()  # idempotent: creates tables / runs migrations
        yield saver
