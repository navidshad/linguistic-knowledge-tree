"""Learner endpoints: list the built-in learners + per-learner status."""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, HTTPException

from klg_ai.activation import get_activation, list_learners
from klg_ai.status import compute_status

from ..deps import get_graph
from ..schemas import LearnerProfileOut, LearnerStatusOut

router = APIRouter(prefix="/api", tags=["learner"])


@router.get("/learners", response_model=list[LearnerProfileOut])
def learners() -> list[LearnerProfileOut]:
    return [
        LearnerProfileOut(id=p.id, label=p.label, description=p.description)
        for p in list_learners()
    ]


@router.get("/learner/{learner_id}/status", response_model=LearnerStatusOut)
def learner_status(learner_id: str) -> LearnerStatusOut:
    activated = get_activation(learner_id)
    if activated is None:
        raise HTTPException(status_code=404, detail=f"Unknown learner '{learner_id}'")
    statuses = {n: s.value for n, s in compute_status(get_graph(), activated).items()}
    return LearnerStatusOut(
        learner_id=learner_id,
        counts=dict(Counter(statuses.values())),
        statuses=statuses,
    )
