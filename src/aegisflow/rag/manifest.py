"""Reproducibility metadata attached to every saved evaluation report."""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from urllib.parse import urlparse

from .qa_config import DEFAULT_QA_CONFIG


KEY_LIBRARIES = (
    "aegisflow",
    "langgraph",
    "langgraph-checkpoint-postgres",
    "psycopg",
    "fastapi",
    "pydantic",
    "httpx",
    "sentence-transformers",
)


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def make_run_manifest(*, cwd: str | None = None) -> dict:
    """Return code, runtime and model configuration for a single evaluation run."""
    git = {"sha": None, "dirty": None}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        git = {"sha": sha, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        pass

    base_url = os.getenv("AEGIS_BASE_URL", "https://api.groq.com/openai/v1")
    return {
        **git,
        "model": os.getenv("AEGIS_MODEL", "openai/gpt-oss-20b"),
        "qa_config_version": DEFAULT_QA_CONFIG.version,
        "base_url_host": urlparse(base_url).hostname,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "temperature": DEFAULT_QA_CONFIG.temperature,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "libraries": {name: _version(name) for name in KEY_LIBRARIES},
    }


def attach_run_manifest(report: dict, *, cwd: str | None = None) -> dict:
    """Return a report with required provenance metadata, rejecting omissions."""
    manifest = report.get("manifest")
    if manifest is None:
        report = {**report, "manifest": make_run_manifest(cwd=cwd)}
        manifest = report["manifest"]
    validate_run_manifest(manifest)
    return report


def validate_run_manifest(manifest: dict) -> None:
    """Raise if a saved evaluation report lacks reproducibility metadata."""
    required = {
        "sha", "dirty", "model", "base_url_host", "date_utc", "temperature",
        "python_version", "libraries", "qa_config_version",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"evaluation report manifest missing fields: {sorted(missing)}")
