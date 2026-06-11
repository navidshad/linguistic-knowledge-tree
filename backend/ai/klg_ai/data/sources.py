"""Data-source registry — the seam for training/evaluating across datasets.

Everything downstream of the loader consumes ``list[LearnerData]``, so adding an
open dataset (EdNet later, etc.) is just registering a new loader here — the
global trainer (``tuning.train_prop``) and the eval harness are written once and
run across any registered source. SLAM is the only source today.
"""
from __future__ import annotations

from collections.abc import Callable

from klg_ai.data.dataset import LearnerData, load_track

# name -> loader(...) -> list[LearnerData]
DATA_SOURCES: dict[str, Callable[..., list[LearnerData]]] = {
    "slam": load_track,
}


def load_source(name: str, *args, **kwargs) -> list[LearnerData]:
    """Load a registered data source into per-learner data for the harness."""
    if name not in DATA_SOURCES:
        have = ", ".join(sorted(DATA_SOURCES))
        raise ValueError(f"unknown data source {name!r} (registered: {have})")
    return DATA_SOURCES[name](*args, **kwargs)
