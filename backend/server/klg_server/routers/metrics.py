"""Validation metrics endpoint (RQ1/RQ3/RQ4/RQ5).

Serves the results JSON produced by ``python -m klg_ai.eval.run`` (the SLAM
ablation table + ROC curves) for the viewer's validation dashboard. Resolves the
file from ``$KLG_METRICS_PATH``, then fresh local runs most-complete first
(``--train-prop`` superset → ``--kgt`` superset → base ``results.json``), then the
committed artifacts (``docs/trained-prop`` → ``docs/phase7`` → ``docs/phase5``).
Each variant file is a strict superset of the previous, so the dashboard works
straight after a checkout and shows the latest local run when one exists.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..schemas import MetricsOut

router = APIRouter(prefix="/api", tags=["metrics"])

_REPO = Path(__file__).resolve().parents[4]


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("KLG_METRICS_PATH")
    if env:
        paths.append(Path(env))
    paths.append(_REPO / "data" / "eval" / "results_trainprop.json")
    paths.append(_REPO / "data" / "eval" / "results_kgt.json")
    paths.append(_REPO / "data" / "eval" / "results.json")
    paths.append(_REPO / "docs" / "trained-prop" / "results.json")
    paths.append(_REPO / "docs" / "phase7" / "results.json")
    paths.append(_REPO / "docs" / "phase5" / "results.json")
    return paths


@router.get("/metrics", response_model=MetricsOut)
def metrics() -> MetricsOut:
    for path in _candidate_paths():
        if path.exists():
            return MetricsOut(**json.loads(path.read_text(encoding="utf-8")))
    raise HTTPException(
        status_code=404,
        detail="No validation results found. Run: python -m klg_ai.eval.run",
    )
