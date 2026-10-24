"""Read Markdown and reStructuredText documents from a documentation tree."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator


_DOC_EXTENSIONS = {".md", ".mdx", ".rst"}
_NOISY_NAME = re.compile(r"(?:^|[-_. ])(?:changelog|release(?:[-_ ]notes?)?)(?:$|[-_. ])", re.I)
_NOISY_DIR = re.compile(r"^(?:changelogs?|releases?|release[-_ ]notes?)$", re.I)
_LANGUAGE_DIR = re.compile(r"^[a-z]{2}(?:[-_][A-Za-z0-9]+)?$")


def iter_documents(docs_dir: str | Path, source: str) -> Iterator[tuple[str, str, str, str]]:
    """Yield ``(doc_id, source, relative_path, raw_text)`` in path order.

    Markdown (including MDX) and reStructuredText files are included.
    Language-code directories (translations), changelogs, and release notes
    are skipped.
    ``relative_path`` is relative to ``docs_dir`` and retains its extension;
    ``doc_id`` omits the extension.
    """
    root = Path(docs_dir)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _DOC_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        if any(_LANGUAGE_DIR.fullmatch(part) for part in parts[:-1]):
            continue
        if any(_NOISY_DIR.fullmatch(part) for part in parts[:-1]):
            continue
        if any(_NOISY_NAME.search(part) for part in parts):
            continue
        relative_path = relative.as_posix()
        doc_id = f"{source}/{relative.with_suffix('').as_posix()}"
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        yield doc_id, source, relative_path, raw_text
