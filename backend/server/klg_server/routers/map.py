"""GET /api/map — the static syntax map (same for every learner)."""
from __future__ import annotations

from fastapi import APIRouter

from ..deps import get_map
from ..schemas import CategoryOut, EdgeOut, MapOut, NodeOut

router = APIRouter(prefix="/api", tags=["map"])


@router.get("/map", response_model=MapOut)
def read_map() -> MapOut:
    m = get_map()
    return MapOut(
        meta=m.meta,
        categories=[CategoryOut(id=c["id"], label=c["label"]) for c in m.categories],
        nodes=[
            NodeOut(id=n.id, label=n.label, category=n.category, cefr=n.cefr, description=n.description)
            for n in m.nodes
        ],
        edges=[EdgeOut(source=e.source, target=e.target, type=e.type) for e in m.edges],
    )
