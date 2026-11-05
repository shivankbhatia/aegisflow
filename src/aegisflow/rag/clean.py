"""Clean Markdown documents into heading-aware text sections."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^(#{1,4})[ \t]+(.+?)\s*#*\s*$")
_INCLUDE = re.compile(r"\{![^\n]*?!\}")
_COMMENT_START = "<!--"
_COMMENT_END = "-->"


@dataclass
class Section:
    """A body of document text and its heading hierarchy."""

    heading_path: list[str]
    body: str


def clean_markdown(raw_text: str, filename: str | Path = "document.md") -> tuple[str, list[Section]]:
    """Return a document title and cleaned sections from raw Markdown.

    Headings through level four start new sections. Fenced code is preserved
    and ignored by the heading scanner. The optional filename supplies a title
    when the document has no headings. Removed MkDocs include directives are
    reported through this module's logger.
    """
    lines = raw_text.splitlines()
    # Front matter is recognized only at the very beginning of the document.
    if lines and lines[0].strip() == "---":
        for end in range(1, len(lines)):
            if lines[end].strip() in {"---", "..."}:
                lines = lines[end + 1 :]
                break

    cleaned_lines: list[str] = []
    in_fence = False
    in_comment = False
    include_count = 0
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            cleaned_lines.append(line)
            continue
        if in_fence:
            cleaned_lines.append(line)
            continue

        remaining = line
        while True:
            if in_comment:
                end = remaining.find(_COMMENT_END)
                if end < 0:
                    remaining = ""
                    break
                remaining = remaining[end + len(_COMMENT_END) :]
                in_comment = False
            start = remaining.find(_COMMENT_START)
            if start < 0:
                break
            end = remaining.find(_COMMENT_END, start + len(_COMMENT_START))
            if end < 0:
                remaining = remaining[:start]
                in_comment = True
                break
            remaining = remaining[:start] + remaining[end + len(_COMMENT_END) :]

        without_includes, removed = _INCLUDE.subn("", remaining)
        include_count += removed
        cleaned_lines.append(without_includes)

    if include_count:
        logger.info("Removed %d MkDocs include directive(s)", include_count)

    fallback_title = Path(filename).stem.replace("-", " ").replace("_", " ").strip() or "document"
    title = fallback_title
    title_from_heading = False
    heading_stack: list[str] = []
    sections: list[Section] = []
    current_body: list[str] = []
    in_fence = False

    def flush() -> None:
        if heading_stack or current_body:
            sections.append(Section(heading_stack.copy(), "\n".join(current_body).strip()))
            current_body.clear()

    for line in cleaned_lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            current_body.append(line)
            continue
        heading = None if in_fence else _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            heading_text = heading.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(heading_text)
            if level == 1 and not title_from_heading:
                title = heading_text
                title_from_heading = True
            continue
        current_body.append(line)
    flush()

    # An empty document still has one predictable section.
    if not sections:
        sections.append(Section([], ""))
    return title, sections
