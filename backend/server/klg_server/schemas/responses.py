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


# --- Phase 5: validation metrics (Duolingo SLAM) ---------------------------

class MetricSet(BaseModel):
    n: int
    base_rate: float
    auroc: float
    F1: float
    accuracy: float
    avglogloss: float


class ModelResultOut(BaseModel):
    name: str
    group: str                        # RQ grouping / baseline tag
    label: str
    metrics: MetricSet
    metrics_cold: MetricSet | None = None   # cold-node slice (RQ3), if any
    roc: list[dict[str, float]]             # [{fpr, tpr}, ...] for plotting


class MetricsDatasetOut(BaseModel):
    source: str
    course: str
    split: str
    n_learners: int
    n_eval_instances: int
    n_cold_instances: int = 0
    mistake_base_rate: float
    node_coverage: float


class MetricsOut(BaseModel):
    """Phase 5 validation results — the ablation table + ROC curves for the viewer."""
    dataset: MetricsDatasetOut
    rqs: dict[str, list[str]]
    models: list[ModelResultOut]
    meta: dict | None = None


# --- Phase 6: Gemini chat demo (dialog turns activate the map) -------------

class NodeEvidenceOut(BaseModel):
    """Which chat turn(s) activated a node, and the mapper's confidence."""
    node_id: str
    confidence: float
    turn_indices: list[int]


class ChatOut(BaseModel):
    """Tutor reply + the knowledge state the learner's dialog turns imply."""
    reply: str
    mapped_nodes: list[str]            # nodes the latest user turn lit up
    confidences: dict[str, float]      # node_id -> cosine, for that latest turn
    counts: dict[str, int]
    statuses: dict[str, str]           # node_id -> status (live overlay)
    mastery: dict[str, float]          # node_id -> mastery [0, 1]
    evidence: list[NodeEvidenceOut]    # per node, the turns behind it (6-B)
