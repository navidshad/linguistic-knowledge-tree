"""GET /api/learner/{id}/status — per-learner node statuses over the map."""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, HTTPException

from klg_ai.activation import get_activation
from klg_ai.status import compute_status

from ..deps import get_graph
from ..schemas import LearnerStatusOut

router = APIRouter(prefix="/api/learner", tags=["learner"])


@router.get("/{learner_id}/status", response_model=LearnerStatusOut)
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
