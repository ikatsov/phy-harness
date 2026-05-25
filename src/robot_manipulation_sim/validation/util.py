"""Shared helpers for validation (dotenv, repo root)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Parse booleans from YAML/CLI (handles ``bool``, numbers, and strings like ``\"false\"``).

    Plain ``bool(x)`` is wrong for strings: ``bool(\"false\")`` is True in Python.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "0", "false", "no", "off", "n", "f"):
            return False
        if s in ("1", "true", "yes", "on", "y", "t"):
            return True
        return default
    return bool(value)


def repo_root() -> Path:
    """Repository root (parent of ``src``)."""
    return Path(__file__).resolve().parents[3]


def load_dotenv_repo() -> None:
    """Load repo-root ``.env`` / ``.env.local`` regardless of cwd (optional ``python-dotenv``)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = repo_root()
    load_dotenv(root / ".env")
    load_dotenv(root / ".env.local")
