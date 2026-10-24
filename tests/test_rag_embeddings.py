import asyncio
import math

from aegisflow.rag.embeddings import BGEEmbedder, QUERY_PREFIX


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [[3.0] + [4.0] + [0.0] * 382 for _ in texts]


def test_bge_embedder_prefixes_queries_batches_and_normalizes_vectors():
    model = FakeModel()
    embedder = BGEEmbedder(model=model, batch_size=32)

    documents = embedder.embed_documents(["passage one", "passage two"])
    query = embedder.embed_query("where is the task?")

    assert model.calls[0][0] == ["passage one", "passage two"]
    assert model.calls[0][1]["batch_size"] == 32
    assert model.calls[1][0] == [QUERY_PREFIX + "where is the task?"]
    assert len(documents[0]) == len(query) == 384
    assert math.isclose(sum(value * value for value in query), 1.0)


def test_bge_async_embedding_wrapper_returns_vectors():
    embedder = BGEEmbedder(model=FakeModel())

    result = asyncio.run(embedder.aembed_query("question"))

    assert len(result) == 384
