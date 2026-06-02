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
