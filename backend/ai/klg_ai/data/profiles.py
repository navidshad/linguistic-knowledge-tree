"""Persistent learner profiles — one JSON file per profile.

The engine is otherwise a pure function of its event stream; this module is the
single place that *stores* a learner's evidence (and chat transcript) so a
profile created in the UI survives a restart. It is plain stdlib file I/O (no
web deps), analogous to ``adapters/synthetic.py`` — the server's CRUD router
calls in here, never the reverse (dependency direction ``server -> klg_ai``).

Config (read per call, so tests can point it at a tmp dir):
* ``KLG_PROFILE_DIR`` — storage dir (default ``<repo>/data/profiles``, git-ignored).

On-disk shape (``<dir>/<id>.json``)::

    {
      "id": "navid-3f1a", "label": "Navid", "description": "...",
      "created_at": "2026-...", "seed_level": "A1" | null,
      "events": [{"node_ids": [...], "correct": true, "ts": 0.0, "source": "review"}],
      "conversation": [{"role": "user"|"tutor", "text": "..."}]
    }

``learner_id`` is *not* stored per event (it is the file's own id) and is
reattached when materializing ``Event`` objects.

Determinism: ``append_events`` ignores caller-supplied ``ts`` and assigns its own
on a monotonic counter (``max(existing) + 1.0``), so repeated appends are
reproducible regardless of wall-clock — what keeps the tests deterministic.
``created_at`` is wall-clock but purely cosmetic (never asserted).
"""
from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from klg_ai.data.adapters.synthetic import generate_events
from klg_ai.core.events import Event
from klg_ai.core.loader import DEFAULT_MAP_PATH

# Built-in synthetic learner ids + the ephemeral ids used by the chat/what-if
# endpoints: a user profile may not be created with (or shadow) any of these.
RESERVED_IDS: frozenset[str] = frozenset(
    {"demo", "beginner", "intermediate", "struggling", "chat", "custom"}
)

_REPO_ROOT = DEFAULT_MAP_PATH.resolve().parents[1]  # <root>/map/<file> -> <root>


@dataclass(frozen=True)
class ProfileMeta:
    """The listing view of a profile (no event/conversation payload)."""
    id: str
    label: str
    description: str = ""


def profile_dir() -> Path:
    return Path(os.environ.get("KLG_PROFILE_DIR") or (_REPO_ROOT / "data" / "profiles"))


def _path(profile_id: str) -> Path:
    return profile_dir() / f"{profile_id}.json"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32]
    return s or "profile"


def _new_id(name: str, existing: set[str]) -> str:
    base = _slug(name)
    for _ in range(1000):
        cand = f"{base}-{secrets.token_hex(2)}"
        if cand not in existing and cand not in RESERVED_IDS:
            return cand
    raise RuntimeError("could not allocate a unique profile id")


def _event_to_json(e: Event) -> dict:
    return {"node_ids": list(e.node_ids), "correct": e.correct, "ts": e.ts, "source": e.source}


def _event_from_json(d: dict, learner_id: str) -> Event:
    return Event(
        learner_id=learner_id,
        node_ids=tuple(d.get("node_ids", ())),
        correct=bool(d.get("correct", True)),
        ts=d.get("ts"),
        source=d.get("source", "review"),
    )


def _read_raw(profile_id: str) -> dict | None:
    """The stored doc, or None if missing/corrupt."""
    path = _path(profile_id)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) and doc.get("id") == profile_id else None
    except (json.JSONDecodeError, OSError):
        return None


def _write(profile_id: str, doc: dict) -> None:
    """Atomic write (temp file + os.replace) so a crash can't leave a torn file."""
    path = _path(profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def list_profiles() -> list[ProfileMeta]:
    """All stored profiles (built-ins are not here — they live in activation.py)."""
    d = profile_dir()
    if not d.exists():
        return []
    out: list[tuple[str, ProfileMeta]] = []
    for path in d.glob("*.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # skip corrupt files
        if isinstance(doc, dict) and doc.get("id"):
            out.append((doc.get("created_at", ""), ProfileMeta(
                id=doc["id"], label=doc.get("label", doc["id"]),
                description=doc.get("description", ""))))
    out.sort(key=lambda t: t[0])  # oldest first (stable order in the picker)
    return [m for _, m in out]


def load_profile(profile_id: str) -> dict | None:
    return _read_raw(profile_id)


def load_events(profile_id: str) -> list[Event] | None:
    """The profile's event stream — the fallback ``_events_for`` resolves to."""
    doc = _read_raw(profile_id)
    if doc is None:
        return None
    return [_event_from_json(d, profile_id) for d in doc.get("events", [])]


def load_conversation(profile_id: str) -> list[dict] | None:
    doc = _read_raw(profile_id)
    if doc is None:
        return None
    convo = doc.get("conversation", [])
    return convo if isinstance(convo, list) else []


def create_profile(
    name: str, *, seed_level: str | None = None, seed_nodes: Sequence[str] = ()
) -> dict:
    """Create (and persist) a new profile; optionally seed review evidence.

    ``seed_nodes`` (computed by the caller, which owns the graph) get
    deterministic correct reviews via ``generate_events`` so a freshly created
    profile can already "know" a CEFR band.
    """
    existing = {m.id for m in list_profiles()}
    pid = _new_id(name, existing)
    desc = f"Custom profile · seeded {seed_level}" if seed_level else "Custom profile"
    events: list[Event] = []
    if seed_nodes:
        events = generate_events(seed_nodes, learner_id=pid, seed=0,
                                 reviews_per_node=3, accuracy=1.0, span_days=14.0)
    doc = {
        "id": pid,
        "label": name.strip() or pid,
        "description": desc,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed_level": seed_level,
        "events": [_event_to_json(e) for e in events],
        "conversation": [],
    }
    _write(pid, doc)
    return doc


def delete_profile(profile_id: str) -> bool:
    """Delete a stored profile; False if it's reserved or doesn't exist."""
    if profile_id in RESERVED_IDS:
        return False
    path = _path(profile_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def append_events(profile_id: str, events: Sequence[Event]) -> list[Event]:
    """Append evidence, restamping ``ts`` on a monotonic counter (max + 1.0).

    Raises ``KeyError`` if the profile doesn't exist.
    """
    doc = _read_raw(profile_id)
    if doc is None:
        raise KeyError(profile_id)
    stored: list[dict] = doc.setdefault("events", [])
    next_ts = max((e["ts"] for e in stored if e.get("ts") is not None), default=-1.0) + 1.0
    stamped: list[Event] = []
    for e in events:
        je = _event_to_json(e)
        je["ts"] = next_ts
        stored.append(je)
        stamped.append(_event_from_json(je, profile_id))
        next_ts += 1.0
    _write(profile_id, doc)
    return stamped


def save_conversation(profile_id: str, conversation: Sequence[dict]) -> None:
    """Persist the full chat transcript so reopening a profile resumes it."""
    doc = _read_raw(profile_id)
    if doc is None:
        raise KeyError(profile_id)
    doc["conversation"] = [{"role": m["role"], "text": m["text"]} for m in conversation]
    _write(profile_id, doc)
