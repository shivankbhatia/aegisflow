import logging

from aegisflow.rag.clean import clean_markdown


def test_python_comment_inside_fence_is_not_a_heading():
    title, sections = clean_markdown(
        "# Example\n\n```python\n# comment\nprint('ok')\n```\n"
    )

    assert title == "Example"
    assert [section.heading_path for section in sections] == [["Example"]]
    assert "# comment" in sections[0].body


def test_heading_paths_nest_and_reset_after_lower_heading():
    _, sections = clean_markdown(
        "# Tutorial\n## Background Tasks\n### Using BackgroundTasks\n"
        "Details.\n## Another Task\nMore details.\n"
    )

    assert [section.heading_path for section in sections] == [
        ["Tutorial"],
        ["Tutorial", "Background Tasks"],
        ["Tutorial", "Background Tasks", "Using BackgroundTasks"],
        ["Tutorial", "Another Task"],
    ]


def test_front_matter_comments_and_includes_are_removed(caplog):
    raw = (
        "---\ntitle: Hidden\n---\n# Visible\n<!-- remove me -->\n"
        "{!../../docs_src/example!}\nKeep this.\n"
    )

    with caplog.at_level(logging.INFO, logger="aegisflow.rag.clean"):
        title, sections = clean_markdown(raw)

    assert title == "Visible"
    assert "remove me" not in sections[0].body
    assert "docs_src" not in sections[0].body
    assert "Keep this." in sections[0].body
    assert "Removed 1 MkDocs include directive(s)" in caplog.text


def test_document_without_headings_uses_filename_title():
    title, sections = clean_markdown("Plain text.\n", "background-tasks.md")

    assert title == "background tasks"
    assert len(sections) == 1
    assert sections[0].heading_path == []
    assert sections[0].body == "Plain text."
