"""Small SQLite-backed, idempotent document chunk store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

from aegisflow.rag.types import Chunk


class SQLiteChunkStore:
    """Persist raw-document hashes and chunks, replacing only changed docs."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(str(database))
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                raw_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def ingest_document(
        self,
        doc_id: str,
        raw_text: str,
        build_chunks: Callable[[], list[Chunk]],
    ) -> bool:
        """Return True if rows were written; skip unchanged raw document text."""
        digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT raw_sha256 FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row is not None and row[0] == digest:
                connection.commit()
                return False

            chunks = build_chunks()
            connection.execute(
                "INSERT INTO documents(doc_id, raw_sha256) VALUES (?, ?) "
                "ON CONFLICT(doc_id) DO UPDATE SET raw_sha256 = excluded.raw_sha256",
                (doc_id, digest),
            )
            connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            connection.executemany(
                "INSERT INTO chunks(id, doc_id, text, metadata_json) VALUES (?, ?, ?, ?)",
                [
                    (c.id, c.doc_id, c.text, json.dumps(c.metadata, sort_keys=True))
                    for c in chunks
                ],
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise

    def row_count(self) -> int:
        """Return the stored chunk row count."""
        return self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self.connection.close()
