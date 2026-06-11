"""Profile CRUD — create/list/delete persistent learners, append evidence.

A profile is a file-backed learner (see ``klg_ai.profiles``): its event log and
chat transcript persist across restarts. Reads (status, timeline, retrain, chat)
go through the existing learner endpoints, which resolve a profile id via
``_events_for``; this router owns the *mutations*. The built-in synthetic
learners are read-only — create/delete/append on a reserved id is rejected.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from klg_ai.activation import _cefr_nodes
from klg_ai.events import Event
from klg_ai.loader import CEFR_LEVELS
from klg_ai.profiles import (
    RESERVED_IDS,
    append_events,
    create_profile,
    delete_profile,
    list_profiles,
    load_conversation,
)

from ..schemas import (
    ConversationOut,
    LearnerProfileOut,
    LearnerStatusOut,
    ProfileCreateIn,
    ProfileEventIn,
)
from .learner import status_payload

router = APIRouter(prefix="/api", tags=["profiles"])


def _levels_up_to(level: str) -> tuple[str, ...]:
    """Every CEFR band at or below ``level`` (a learner at B1 likely knows A1–B1)."""
    return CEFR_LEVELS[: CEFR_LEVELS.index(level) + 1]


@router.get("/profiles", response_model=list[LearnerProfileOut])
def profiles() -> list[LearnerProfileOut]:
    """User-created profiles (editable). Built-ins are under ``GET /api/learners``."""
    return [
        LearnerProfileOut(id=p.id, label=p.label, description=p.description, editable=True)
        for p in list_profiles()
    ]


@router.post("/profiles", response_model=LearnerProfileOut, status_code=201)
def create(body: ProfileCreateIn) -> LearnerProfileOut:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Profile name is required")
    seed_nodes = _cefr_nodes(*_levels_up_to(body.seed_level)) if body.seed_level else ()
    doc = create_profile(name, seed_level=body.seed_level, seed_nodes=seed_nodes)
    return LearnerProfileOut(id=doc["id"], label=doc["label"],
                             description=doc["description"], editable=True)


@router.delete("/profiles/{profile_id}", status_code=204)
def remove(profile_id: str) -> None:
    if profile_id in RESERVED_IDS:
        raise HTTPException(status_code=409, detail="Built-in profiles are read-only")
    if not delete_profile(profile_id):
        raise HTTPException(status_code=404, detail=f"Unknown profile '{profile_id}'")


@router.post("/profiles/{profile_id}/events", response_model=LearnerStatusOut)
def add_event(profile_id: str, body: ProfileEventIn, kgt: bool = False) -> LearnerStatusOut:
    """Append one piece of evidence (e.g. marking a node known) and return new status."""
    if profile_id in RESERVED_IDS:
        raise HTTPException(status_code=409, detail="Built-in profiles are read-only")
    try:
        append_events(profile_id, [Event(
            learner_id=profile_id, node_ids=tuple(body.node_ids),
            correct=body.correct, source=body.source)])
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown profile '{profile_id}'")
    payload = status_payload(profile_id, kgt)
    assert payload is not None  # just appended -> resolvable
    return payload


@router.get("/profiles/{profile_id}/conversation", response_model=ConversationOut)
def conversation(profile_id: str) -> ConversationOut:
    convo = load_conversation(profile_id)
    if convo is None:
        raise HTTPException(status_code=404, detail=f"Unknown profile '{profile_id}'")
    return ConversationOut(profile_id=profile_id, messages=convo)
