from aegisflow.rag.corpus import iter_documents


def test_iter_documents_yields_stable_ids_and_skips_non_docs(tmp_path):
    (tmp_path / "tutorial").mkdir()
    (tmp_path / "tutorial" / "background-tasks.md").write_text("tasks", encoding="utf-8")
    (tmp_path / "index.rst").write_text("index", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not text")

    documents = list(iter_documents(tmp_path, "fastapi"))

    assert documents == [
        ("fastapi/index", "fastapi", "index.rst", "index"),
        (
            "fastapi/tutorial/background-tasks",
            "fastapi",
            "tutorial/background-tasks.md",
            "tasks",
        ),
    ]
