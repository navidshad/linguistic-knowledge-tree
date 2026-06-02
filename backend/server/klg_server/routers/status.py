"""POST /api/status — compute statuses for an arbitrary activated set.

Powers interactive what-if exploration in the viewer (toggle nodes on/off) while
keeping the status/gap logic in the engine, not duplicated in the frontend.
"""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter

from klg_ai.status import compute_status

from ..deps import get_graph
from ..schemas import ActivationIn, LearnerStatusOut

router = APIRouter(prefix="/api", tags=["status"])


@router.post("/status", response_model=LearnerStatusOut)
def status_for(body: ActivationIn) -> LearnerStatusOut:
    statuses = {n: s.value for n, s in compute_status(get_graph(), set(body.activated)).items()}
    return LearnerStatusOut(
        learner_id="custom",
        counts=dict(Counter(statuses.values())),
        statuses=statuses,
    )
