"""Token-aware chunking strategies for cleaned RAG documents."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from aegisflow.rag.clean import Section
from aegisflow.rag.types import Chunk


TokenCounter = Callable[[str], int]
_DEFAULT_ENCODER = None


def _encoder():
    global _DEFAULT_ENCODER
    if _DEFAULT_ENCODER is None:
        import tiktoken

        _DEFAULT_ENCODER = tiktoken.get_encoding("o200k_base")
    return _DEFAULT_ENCODER


def _count(text: str, token_counter: TokenCounter | None) -> int:
    return token_counter(text) if token_counter else len(_encode(text))


def _encode(text: str) -> list[int]:
    # Documentation may mention strings that resemble special tokens; those
    # should be measured as literal text, not rejected by tiktoken.
    return _encoder().encode(text, disallowed_special=())


def _make_chunk(
    doc_id: str,
    index: int,
    text: str,
    *,
    source: str,
    title: str,
    heading_path: list[str],
    strategy: str,
    token_counter: TokenCounter | None,
    extra_metadata: dict | None = None,
) -> Chunk:
    metadata = {
        "source": source,
        "title": title,
        "heading_path": heading_path,
        "chunk_strategy": strategy,
        "token_count": _count(text, token_counter),
    }
    metadata.update(extra_metadata or {})
    return Chunk(f"{doc_id}#{index}", doc_id, text, metadata)


def _section_text(section: Section) -> str:
    return "\n".join([*section.heading_path, section.body]).strip()


def fixed_size_chunks(
    sections: Sequence[Section],
    *,
    doc_id: str,
    source: str,
    title: str,
    token_counter: TokenCounter | None = None,
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[Chunk]:
    """Make overlapping, fixed-token windows from a document's flattened text."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    full_text = "\n\n".join(text for s in sections if (text := _section_text(s)))
    token_ids = _encode(full_text)
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for index, start in enumerate(range(0, len(token_ids), step)):
        end = min(start + chunk_size, len(token_ids))
        text = _encoder().decode(token_ids[start:end])
        chunks.append(
            _make_chunk(
                doc_id,
                index,
                text,
                source=source,
                title=title,
                heading_path=[],
                strategy="fixed-size",
                token_counter=token_counter,
                extra_metadata={"token_start": start, "token_end": end},
            )
        )
    return chunks


def _paragraph_blocks(text: str) -> list[tuple[str, bool]]:
    """Return paragraph blocks, keeping each fenced code block as one block."""
    lines = text.splitlines()
    blocks: list[tuple[str, bool]] = []
    ordinary: list[str] = []
    code: list[str] = []
    in_fence = False

    def flush_ordinary() -> None:
        if ordinary:
            value = "\n".join(ordinary).strip()
            if value:
                blocks.append((value, False))
            ordinary.clear()

    for line in lines:
        if line.lstrip().startswith("```"):
            if not in_fence:
                flush_ordinary()
                code = [line]
                in_fence = True
            else:
                code.append(line)
                blocks.append(("\n".join(code), True))
                code = []
                in_fence = False
            continue
        if in_fence:
            code.append(line)
        elif not line.strip():
            flush_ordinary()
        else:
            ordinary.append(line)
    if in_fence:
        # Keep malformed, unclosed fences intact rather than manufacturing a split.
        blocks.append(("\n".join(code), True))
    flush_ordinary()
    return blocks


def _split_long_paragraph(text: str, max_tokens: int, token_counter: TokenCounter | None) -> list[str]:
    """Split a long prose block by lines, then token windows as a final fallback."""
    lines = text.splitlines()
    parts: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = "\n".join([*current, line]).strip()
        if current and _count(candidate, token_counter) > max_tokens:
            parts.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current).strip())
    if any(_count(part, token_counter) > max_tokens for part in parts):
        ids = _encode(text)
        parts = [
            _encoder().decode(ids[start : start + max_tokens])
            for start in range(0, len(ids), max_tokens)
        ]
    return parts


def _split_oversized_code(
    block: str,
    *,
    prefix: str,
    max_tokens: int,
    token_counter: TokenCounter | None,
) -> list[str]:
    """Re-fence pieces of a code block that cannot fit atomically under the cap."""
    lines = block.splitlines()
    opener = lines[0]
    has_closer = len(lines) > 1 and lines[-1].lstrip().startswith("```")
    closer = lines[-1] if has_closer else "```"
    body = "\n".join(lines[1:-1] if has_closer else lines[1:])
    body_ids = _encode(body)
    chunks: list[str] = []
    position = 0
    while position < len(body_ids) or (not body_ids and not chunks):
        end = min(len(body_ids), position + max_tokens)
        while end > position:
            decoded = _encoder().decode(body_ids[position:end])
            candidate = f"{prefix}\n\n{opener}\n{decoded}\n{closer}".strip() if prefix else f"{opener}\n{decoded}\n{closer}"
            if _count(candidate, token_counter) <= max_tokens:
                break
            end -= 1
        if end == position and body_ids:
            raise ValueError("Heading and code fences leave no room under the chunk token limit")
        if not body_ids:
            candidate = f"{prefix}\n\n{opener}\n{closer}".strip() if prefix else f"{opener}\n{closer}"
            if _count(candidate, token_counter) > max_tokens:
                raise ValueError("Heading and code fences exceed the chunk token limit")
            chunks.append(candidate)
            break
        chunks.append(candidate)
        position = end
    return chunks


def _split_section_body(
    section: Section,
    *,
    max_tokens: int,
    token_counter: TokenCounter | None,
) -> list[str]:
    prefix = " > ".join(section.heading_path)
    blocks = _paragraph_blocks(section.body)
    if not blocks:
        return [prefix] if prefix else ([""] if not section.body.strip() else [section.body.strip()])

    results: list[str] = []
    current: list[str] = []

    def render(blocks_: list[str]) -> str:
        body = "\n\n".join(blocks_).strip()
        return f"{prefix}\n\n{body}".strip() if prefix else body

    body_token_limit = max_tokens - (_count(prefix + "\n\n", token_counter) if prefix else 0)
    if body_token_limit < 1:
        raise ValueError("Heading path exceeds the chunk token limit")

    def flush() -> None:
        if current:
            results.append(render(current))
            current.clear()

    for block, is_code in blocks:
        candidate = render([*current, block])
        if current and _count(candidate, token_counter) > max_tokens:
            flush()
        if _count(render([block]), token_counter) > max_tokens:
            if is_code:
                for fragment in _split_oversized_code(
                    block,
                    prefix=prefix,
                    max_tokens=max_tokens,
                    token_counter=token_counter,
                ):
                    flush()
                    results.append(fragment)
                continue
            for split_part in _split_long_paragraph(block, body_token_limit, token_counter):
                if _count(render([split_part]), token_counter) > max_tokens:
                    raise ValueError("Heading prefix leaves no room under the chunk token limit")
                flush()
                results.append(render([split_part]))
            continue
        current.append(block)
    flush()
    return results


def structure_aware_chunks(
    sections: Sequence[Section],
    *,
    doc_id: str,
    source: str,
    title: str,
    token_counter: TokenCounter | None = None,
    max_tokens: int = 500,
    merge_below_tokens: int = 80,
) -> list[Chunk]:
    """Chunk by sections, joining small adjacent sections before token splits."""
    if max_tokens < 1 or merge_below_tokens < 0:
        raise ValueError("Token limits must be positive")
    grouped: list[list[Section]] = []
    index = 0
    while index < len(sections):
        group = [sections[index]]
        group_text = _section_text(sections[index])
        depth = len(sections[index].heading_path)
        index += 1
        while (
            _count(group_text, token_counter) < merge_below_tokens
            and index < len(sections)
            and len(sections[index].heading_path) >= depth
        ):
            group.append(sections[index])
            group_text = "\n\n".join(_section_text(s) for s in group).strip()
            index += 1
        grouped.append(group)

    chunks: list[Chunk] = []
    for group in grouped:
        path = group[0].heading_path
        merged_body = "\n\n".join(
            (
                (" > ".join(s.heading_path) + "\n\n")
                if section_index and s.heading_path
                else ""
            )
            + s.body
            for section_index, s in enumerate(group)
        ).strip()
        merged = Section(path, merged_body)
        for text in _split_section_body(
            merged, max_tokens=max_tokens, token_counter=token_counter
        ):
            if not text.strip():
                continue
            chunks.append(
                _make_chunk(
                    doc_id,
                    len(chunks),
                    text,
                    source=source,
                    title=title,
                    heading_path=path,
                    strategy="structure-aware",
                    token_counter=token_counter,
                )
            )
    return chunks


def semantic_chunks(*args, **kwargs) -> list[Chunk]:
    """Semantic chunking is implemented with the 2.2 embedder."""
    raise NotImplementedError("Build semantic chunking after the 2.2 embedder")
