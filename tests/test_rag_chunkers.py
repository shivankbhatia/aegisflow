import tiktoken

from aegisflow.rag.clean import Section
from aegisflow.rag.chunkers import fixed_size_chunks, structure_aware_chunks
from aegisflow.rag.ingest import SQLiteChunkStore
from aegisflow.rag.types import Chunk


ENCODER = tiktoken.get_encoding("o200k_base")


def test_fixed_size_chunks_fit_have_unique_ids_and_round_trip_nonoverlap():
    text = " ".join(f"token{i}" for i in range(1300))
    sections = [Section(["Guide"], text)]
    chunks = fixed_size_chunks(
        sections, doc_id="docs/guide", source="docs", title="Guide"
    )

    assert chunks
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert all(chunk.text and chunk.metadata["token_count"] <= 400 for chunk in chunks)
    assert all(
        {"source", "title", "heading_path", "chunk_strategy", "token_count"}
        <= chunk.metadata.keys()
        for chunk in chunks
    )
    reconstructed = []
    for index, chunk in enumerate(chunks):
        token_start = 0 if index == 0 else 60
        reconstructed.extend(ENCODER.encode(chunk.text)[token_start:])
    assert ENCODER.decode(reconstructed) == "Guide\n" + text


def test_structure_chunks_keep_fences_whole_and_include_metadata():
    code = "```python\n" + "\n".join(f"print({i})" for i in range(700)) + "\n```"
    sections = [
        Section(["Guide", "Examples"], "A short introduction.\n\n" + code),
        Section(["Guide", "Examples", "More"], "More explanation. " * 80),
    ]

    chunks = structure_aware_chunks(
        sections, doc_id="docs/guide", source="docs", title="Guide"
    )

    assert chunks
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk.text
        assert chunk.metadata["token_count"] <= 500
        assert len([line for line in chunk.text.splitlines() if line.lstrip().startswith("```")]) % 2 == 0
        assert chunk.metadata["source"] == "docs"
        assert chunk.metadata["title"] == "Guide"
        assert chunk.metadata["chunk_strategy"] == "structure-aware"


def test_reingest_unchanged_raw_document_keeps_chunk_row_count():
    store = SQLiteChunkStore()
    document = "raw source document"
    chunks = [
        Chunk("docs/one#0", "docs/one", "first chunk", {"source": "docs"}),
        Chunk("docs/one#1", "docs/one", "second chunk", {"source": "docs"}),
    ]
    calls = 0

    def build_chunks():
        nonlocal calls
        calls += 1
        return chunks

    assert store.ingest_document("docs/one", document, build_chunks) is True
    initial_rows = store.row_count()
    assert store.ingest_document("docs/one", document, build_chunks) is False

    assert initial_rows == store.row_count() == 2
    assert calls == 1
    chunks = chunks[:1]
    assert store.ingest_document("docs/one", document + " updated", build_chunks) is True
    assert store.row_count() == 1
    assert calls == 2
    store.close()
