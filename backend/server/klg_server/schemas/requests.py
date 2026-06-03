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
    conversation) keys the per-session pipeline trace.
    """
    messages: list[ChatTurn] = []
    activated: list[str] = []
    session_id: str | None = None
