"""Pydantic request models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ActivationIn(BaseModel):
    """An arbitrary activated node set — for interactive what-if exploration."""
    activated: list[str] = []


class ChatTurn(BaseModel):
    """One message in the tutor conversation."""
    role: Literal["user", "tutor"]
    text: str


class ChatIn(BaseModel):
    """A chat conversation + the learner's current known-set.

    Stateless, mirroring ``ActivationIn``: the client carries the full message
    history and activated set, and the server re-derives the knowledge state from
    the learner's turns each request. ``session_id`` (client-generated, stable per
    conversation) keys the per-session pipeline trace. ``profile_id`` (Phase 8)
    targets a persistent profile: its dialog evidence + transcript are then saved.
    """
    messages: list[ChatTurn] = []
    activated: list[str] = []
    session_id: str | None = None
    profile_id: str | None = None
    # Evidence→node mapper for the learner turns. "semantic" (default) = the
    # validated K-BERT mapper + Gemini grading (Gemini only vetoes); "gemini" =
    # Gemini tags concepts directly (proposes), lifting the recall ceiling.
    tagger: Literal["semantic", "gemini"] | None = None


Cefr = Literal["A1", "A2", "B1", "B2", "C1", "C2"]


class ProfileCreateIn(BaseModel):
    """Create a persistent learner profile; optionally seed a starting CEFR band."""
    name: str
    seed_level: Cefr | None = None


class ProfileEventIn(BaseModel):
    """Append one piece of evidence to a profile (e.g. marking a node known)."""
    node_ids: list[str]
    correct: bool = True
    source: Literal["review", "dialog", "exposure"] = "review"
