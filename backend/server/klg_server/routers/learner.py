"""Learner endpoints: list the built-in learners, per-learner status, timeline.

Phase 7 (RQ5): ``?kgt=1`` on the status endpoint computes mastery over the
learner's *personalized* graph and returns the edge adjustments behind it;
``POST /learner/{id}/retrain`` runs the per-learner gradient comparator live,
epoch by epoch, so the viewer can animate "retraining" next to KGT's one-shot
tuning.
"""
from __future__ import annotations

import time
from collections import Counter

from fastapi import APIRouter, HTTPException

from klg_ai.activation import (
    DEFAULT_CONFIG,
    EngineConfig,
    compute_edge_adjustments,
    compute_mastery,
    learner_events,
    list_learners,
    mastery_timeline,
    threshold_activated,
)
from klg_ai.gaps import gap_scores
from klg_ai.status import compute_status

from ..deps import get_graph
from ..schemas import (
    EdgeAdjustmentOut,
    LearnerProfileOut,
    LearnerStatusOut,
    RetrainEpochOut,
    RetrainOut,
    TimelineFrameOut,
    TimelineOut,
)

router = APIRouter(prefix="/api", tags=["learner"])

MAX_FRAMES = 60  # cap timeline resolution so a request stays a handful of propagations
MAX_RETRAIN_EPOCHS = 100

KGT_CONFIG = EngineConfig(kgt=True)  # engine defaults + per-learner edge tuning


def _round(mastery: dict[str, float], ndigits: int = 3) -> dict[str, float]:
    """Trim mastery to a few decimals — plenty for an opacity overlay, smaller payload."""
    return {n: round(m, ndigits) for n, m in mastery.items()}


def status_payload(learner_id: str, kgt: bool = False) -> LearnerStatusOut | None:
    """Full per-learner status (mastery + gap scores + KGT edges), or None if unknown.

    The single place this payload is built, so file-backed profiles get the
    identical mastery / KGT / gap-score treatment as the built-in learners.
    """
    g = get_graph()
    mastery = compute_mastery(learner_id, config=KGT_CONFIG if kgt else DEFAULT_CONFIG)
    if mastery is None:
        return None
    activated = threshold_activated(mastery)
    statuses = {n: s.value for n, s in compute_status(g, activated).items()}
    adjustments = None
    if kgt:
        adjustments = [
            EdgeAdjustmentOut(**vars(a)) for a in compute_edge_adjustments(learner_id) or []
        ]
    return LearnerStatusOut(
        learner_id=learner_id,
        counts=dict(Counter(statuses.values())),
        statuses=statuses,
        mastery=_round(mastery),
        gap_scores=_round(gap_scores(g, statuses, mastery)),
        edge_adjustments=adjustments,
    )


@router.get("/learners", response_model=list[LearnerProfileOut])
def learners() -> list[LearnerProfileOut]:
    return [
        LearnerProfileOut(id=p.id, label=p.label, description=p.description, editable=p.editable)
        for p in list_learners()
    ]


@router.get("/learner/{learner_id}/status", response_model=LearnerStatusOut)
def learner_status(learner_id: str, kgt: bool = False) -> LearnerStatusOut:
    # ``kgt`` switches to the learner's personalized graph (Phase 7, RQ5) and
    # reports the edge adjustments; the default path is unchanged.
    payload = status_payload(learner_id, kgt)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown learner '{learner_id}'")
    return payload


@router.post("/learner/{learner_id}/retrain", response_model=RetrainOut)
def learner_retrain(learner_id: str, epochs: int = 30) -> RetrainOut:
    """Run the RQ5 'retrain per user' comparator live, epoch by epoch.

    Gradient-fits this learner's edge multipliers on their own events (the
    graph is small enough that this is sub-second) and returns per-epoch loss +
    factor snapshots so the viewer can animate the fit converging — next to the
    wall time of the closed-form KGT rule on the same events. Stateless and
    deterministic; nothing is stored.
    """
    from klg_ai.eval.retrain import fit_edge_factors
    from klg_ai.kgt import tune_edges

    events = learner_events(learner_id)
    if events is None:
        raise HTTPException(status_code=404, detail=f"Unknown learner '{learner_id}'")
    g = get_graph()
    n_epochs = max(1, min(epochs, MAX_RETRAIN_EPOCHS))

    t0 = time.perf_counter()
    trace = fit_edge_factors(g, events, KGT_CONFIG, epochs=n_epochs, record_epochs=True)
    wall_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    kgt_adjustments = tune_edges(g, events, KGT_CONFIG).adjustments  # timed for the side-by-side
    kgt_wall_ms = (time.perf_counter() - t0) * 1000

    return RetrainOut(
        learner_id=learner_id,
        n_items=trace.n_items,
        wall_ms=round(wall_ms, 1),
        kgt_wall_ms=round(kgt_wall_ms, 1),
        epochs=[
            RetrainEpochOut(
                epoch=i + 1,
                loss=round(loss, 4),
                edge_adjustments=_adjustments_out(g, factors),
            )
            for i, (loss, factors) in enumerate(zip(trace.losses, trace.epoch_factors or []))
        ],
    )


def _adjustments_out(g, factors: dict, min_effect: float = 0.1) -> list[EdgeAdjustmentOut]:
    """Gradient-fitted factors -> the same wire shape as KGT adjustments."""
    out: list[EdgeAdjustmentOut] = []
    for (u, v), (m_back, m_fwd) in factors.items():
        dominant = m_back if abs(m_back - 1.0) >= abs(m_fwd - 1.0) else m_fwd
        if abs(dominant - 1.0) < min_effect:
            continue
        kind = "removed" if dominant <= 0.1 else ("weakened" if dominant < 1.0 else "strengthened")
        out.append(EdgeAdjustmentOut(
            source=u, target=v,
            factor_back=round(m_back, 4), factor_fwd=round(m_fwd, 4),
            kind=kind, reason="Gradient-fitted on this learner's history",
        ))
    return out


@router.get("/learner/{learner_id}/timeline", response_model=TimelineOut)
def learner_timeline(learner_id: str, frames: int = 24) -> TimelineOut:
    """Mastery sampled across the learner's history — powers the viewer scrubber."""
    n = max(2, min(frames, MAX_FRAMES))
    timeline = mastery_timeline(learner_id, frames=n)
    if timeline is None:
        raise HTTPException(status_code=404, detail=f"Unknown learner '{learner_id}'")
    g = get_graph()
    out = []
    for f in timeline:
        statuses = {node: s.value for node, s in compute_status(g, f.activated).items()}
        out.append(
            TimelineFrameOut(
                t=round(f.t, 3),
                counts=dict(Counter(statuses.values())),
                statuses=statuses,
                mastery=_round(f.mastery),
            )
        )
    return TimelineOut(
        learner_id=learner_id, t_start=out[0].t, t_end=out[-1].t, frames=out
    )
