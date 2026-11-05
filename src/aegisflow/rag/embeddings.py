"""Local BGE sentence embeddings for dense retrieval."""

from __future__ import annotations

import asyncio
import math
import os
import platform
import threading
from collections.abc import Sequence


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384


class BGEEmbedder:
    """Sentence-transformers wrapper returning normalized 384d vectors."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        batch_size: int = 48,
        device: str | None = None,
        model=None,
        inference_lock: threading.Lock | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if device is None:
            device = self._preferred_device()
        self.batch_size = batch_size
        self._inference_lock = inference_lock or threading.Lock()
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=device)
        self._model = model

    @staticmethod
    def _preferred_device() -> str:
        configured = os.getenv("AEGIS_RAG_DEVICE")
        if configured:
            if configured not in {"mps", "cpu"}:
                raise ValueError("AEGIS_RAG_DEVICE must be 'mps' or 'cpu'")
            if configured == "mps":
                try:
                    import torch
                    if not torch.backends.mps.is_available():
                        raise RuntimeError("AEGIS_RAG_DEVICE=mps but PyTorch MPS is unavailable")
                except ImportError as exc:
                    raise RuntimeError("AEGIS_RAG_DEVICE=mps requires PyTorch") from exc
            return configured
        if platform.system() == "Darwin":
            try:
                import torch

                if torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
        return "cpu"

    @staticmethod
    def _normalize(vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        if len(values) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSIONS}-dimensional embeddings; got {len(values)}"
            )
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            raise ValueError("Embedding model returned a zero vector")
        return [value / norm for value in values]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._inference_lock:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return [self._normalize(vector) for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document passages in batches without adding a query prefix."""
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query using BGE's retrieval instruction prefix."""
        return self._encode([QUERY_PREFIX + text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async wrapper that keeps model inference off the event loop."""
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Async wrapper that keeps model inference off the event loop."""
        return await asyncio.to_thread(self.embed_query, text)
