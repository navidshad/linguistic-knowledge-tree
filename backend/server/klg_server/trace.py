"""Per-session trace of the chat pipeline → a temp JSONL file (one per session).

Each learner turn appends one JSON line tracing its journey through the system:
input → embed + cosine match (nodes + scores) → ``dialog`` event → direct evidence
score → GNN lift → status. Lets you replay exactly what a sentence did to the map.

Config (read per call, so tests can toggle it):
* ``KLG_TRACE``      — set to ``"0"`` to disable (default: on).
* ``KLG_TRACE_DIR``  — output dir (default: ``<tempdir>/klg-chat-traces``).

One file per session id: ``<session>.json`` — a single pretty-printed JSON object
``{"session", "turns": [...]}``, rewritten each turn so it formats/folds in an
editor (unlike append-only JSONL).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path


def enabled() -> bool:
    return os.environ.get("KLG_TRACE", "1") != "0"


def trace_dir() -> Path:
    return Path(os.environ.get("KLG_TRACE_DIR") or (Path(tempfile.gettempdir()) / "klg-chat-traces"))


def _safe(session: str) -> str:
    """Filesystem-safe session id (avoid path traversal / odd chars)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", session or "")[:80] or "default"


def trace_path(session: str) -> Path:
    return trace_dir() / f"{_safe(session)}.json"


def record(session: str, data: dict) -> Path | None:
    """Append one turn to ``session``'s trace; rewrite a single pretty JSON file.

    The file is a valid JSON object ``{"session", "turns": [...]}`` (read-modify-write
    each turn) so an editor can format/fold it. No-op (returns None) when disabled.
    """
    if not enabled():
        return None
    path = trace_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: dict = {"session": session, "turns": []}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("turns"), list):
                doc = existing
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/partial file → start fresh
    doc["turns"].append(data)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
