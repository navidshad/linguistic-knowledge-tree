"""Pydantic response models — the API contract the Vue frontend consumes."""
from __future__ import annotations

from pydantic import BaseModel

from .requests import ChatTurn


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


class EdgeAdjustmentOut(BaseModel):
    """One KGT edge re-weighting in a learner's personal graph (Phase 7, RQ5)."""
    source: str                       # the prerequisite
    target: str                       # the dependent
    factor_back: float                # multiplier on the backward inference message
    factor_fwd: float                 # multiplier on the forward readiness message
    kind: str                         # strengthened | weakened | removed
    reason: str                       # human-readable, cites the learner's evidence


class LearnerStatusOut(BaseModel):
    learner_id: str
    counts: dict[str, int]            # status -> count
    statuses: dict[str, str]          # node_id -> status
    # Continuous per-node mastery [0, 1] behind the discrete status (Phase 4-B
    # confidence overlay). None for the what-if endpoint, which is given a hand-
    # picked activated set with no evidence to score.
    mastery: dict[str, float] | None = None
    # §3.7 recommender handoff: pedagogical priority for interior-gap/frontier
    # nodes (level_weight × (1 − mastery)). None where mastery is None.
    gap_scores: dict[str, float] | None = None
    # Personal-graph deltas (only with ?kgt=1; None otherwise).
    edge_adjustments: list[EdgeAdjustmentOut] | None = None


class LearnerProfileOut(BaseModel):
    id: str
    label: str
    description: str = ""
    editable: bool = False            # built-ins read-only; file-backed profiles editable


class ConversationOut(BaseModel):
    """A profile's persisted chat transcript (so a session resumes)."""
    profile_id: str
    messages: list[ChatTurn]


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


class CostOut(BaseModel):
    """Measured compute cost of one model's fit+predict (RQ5 runs only)."""
    fit_predict_seconds: float
    seconds_per_learner: float


class ModelResultOut(BaseModel):
    name: str
    group: str                        # RQ grouping / baseline tag
    label: str
    metrics: MetricSet
    metrics_cold: MetricSet | None = None   # cold-node slice (RQ3), if any
    roc: list[dict[str, float]]             # [{fpr, tpr}, ...] for plotting
    cost: CostOut | None = None             # present in --kgt (RQ5) runs
    retrain_curve: list[dict[str, float]] | None = None  # [{epoch, loss}] (retrain arm)


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
    """Which chat turn(s) touched a node, and the mapper's confidence."""
    node_id: str
    confidence: float
    turn_indices: list[int]
    incorrect_turn_indices: list[int] = []  # subset graded as wrong usage


class ChatOut(BaseModel):
    """Tutor reply + the knowledge state the learner's dialog turns imply."""
    reply: str
    mapped_nodes: list[str]            # nodes the latest user turn touched (graded as used)
    confidences: dict[str, float]      # node_id -> cosine, for that latest turn
    grades: dict[str, bool] = {}       # node_id -> used correctly?, for that latest turn
    counts: dict[str, int]
    statuses: dict[str, str]           # node_id -> status (live overlay)
    mastery: dict[str, float]          # node_id -> mastery [0, 1]
    evidence: list[NodeEvidenceOut]    # per node, the turns behind it (6-B)
    # Phase 7: KGT live on the conversation — wrong usage weakens inference edges.
    edge_adjustments: list[EdgeAdjustmentOut] | None = None


# --- Phase 7: live per-learner retrain (the RQ5 demo endpoint) --------------

class RetrainEpochOut(BaseModel):
    """One epoch of the gradient fit: train loss + the edge factors so far."""
    epoch: int
    loss: float
    edge_adjustments: list[EdgeAdjustmentOut]


class RetrainOut(BaseModel):
    """A per-learner gradient retrain, epoch by epoch, vs. KGT's one-shot time."""
    learner_id: str
    n_items: int                       # supervision pairs the fit saw
    wall_ms: float                     # full gradient fit
    kgt_wall_ms: float                 # closed-form KGT on the same events
    epochs: list[RetrainEpochOut]
