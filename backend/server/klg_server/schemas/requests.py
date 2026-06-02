"""Pydantic request models."""
from __future__ import annotations

from pydantic import BaseModel


class ActivationIn(BaseModel):
    """An arbitrary activated node set — for interactive what-if exploration."""
    activated: list[str] = []
