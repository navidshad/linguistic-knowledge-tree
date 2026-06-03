"""Pydantic response models — the API contract the Vue frontend consumes."""
from __future__ import annotations

from pydantic import BaseModel


class HealthOut(BaseModel):
    status: str


class CategoryOut(BaseModel):
    id: str
    label: str


class NodeOut(BaseModel):
    id: str
    label: str
    category: str
    cefr: str
    description: str = ""


class EdgeOut(BaseModel):
    source: str
    target: str
    type: str = "prerequisite"


class MapOut(BaseModel):
    meta: dict
    categories: list[CategoryOut]
    nodes: list[NodeOut]
    edges: list[EdgeOut]


class LearnerStatusOut(BaseModel):
    learner_id: str
    counts: dict[str, int]            # status -> count
    statuses: dict[str, str]          # node_id -> status
    # Continuous per-node mastery [0, 1] behind the discrete status (Phase 4-B
    # confidence overlay). None for the what-if endpoint, which is given a hand-
    # picked activated set with no evidence to score.
    mastery: dict[str, float] | None = None


class LearnerProfileOut(BaseModel):
    id: str
    label: str
    description: str = ""


class TimelineFrameOut(BaseModel):
    """A learner's knowledge state at one reference time (the scrubber's frame)."""
    t: float                          # reference time in days on the fixed origin
    counts: dict[str, int]
    statuses: dict[str, str]          # node_id -> status at this instant
    mastery: dict[str, float]         # node_id -> mastery at this instant


class TimelineOut(BaseModel):
    learner_id: str
    t_start: float
    t_end: float
    frames: list[TimelineFrameOut]
