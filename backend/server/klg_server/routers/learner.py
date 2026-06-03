"""Learner endpoints: list the built-in learners, per-learner status, timeline."""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, HTTPException

from klg_ai.activation import (
    compute_mastery,
    list_learners,
    mastery_timeline,
    threshold_activated,
)
from klg_ai.status import compute_status

from ..deps import get_graph
from ..schemas import (
    LearnerProfileOut,
    LearnerStatusOut,
    TimelineFrameOut,
    TimelineOut,
)

router = APIRouter(prefix="/api", tags=["learner"])

MAX_FRAMES = 60  # cap timeline resolution so a request stays a handful of propagations


def _round(mastery: dict[str, float], ndigits: int = 3) -> dict[str, float]:
    """Trim mastery to a few decimals — plenty for an opacity overlay, smaller payload."""
    return {n: round(m, ndigits) for n, m in mastery.items()}


@router.get("/learners", response_model=list[LearnerProfileOut])
def learners() -> list[LearnerProfileOut]:
    return [
        LearnerProfileOut(id=p.id, label=p.label, description=p.description)
        for p in list_learners()
    ]


@router.get("/learner/{learner_id}/status", response_model=LearnerStatusOut)
def learner_status(learner_id: str) -> LearnerStatusOut:
    # One propagation: compute mastery, then derive the known-set from it.
    mastery = compute_mastery(learner_id)
    if mastery is None:
        raise HTTPException(status_code=404, detail=f"Unknown learner '{learner_id}'")
    activated = threshold_activated(mastery)
    statuses = {n: s.value for n, s in compute_status(get_graph(), activated).items()}
    return LearnerStatusOut(
        learner_id=learner_id,
        counts=dict(Counter(statuses.values())),
        statuses=statuses,
        mastery=_round(mastery),
    )


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
