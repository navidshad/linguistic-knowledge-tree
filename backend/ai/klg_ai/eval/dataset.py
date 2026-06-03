"""Load a SLAM track into per-learner train history + evaluation set.

The shared task splits each learner's first ~30 days temporally: ``.train``
(earlier) vs. ``.dev`` / ``.test`` (later). So predicting the eval split is
genuine next-step prediction from prior history — exactly what the engine's
point-in-time mastery is built for.

Each labeled token becomes an ``Interaction`` (timestamped, with its mapped
concept nodes and correctness). A learner's engine ``Event`` stream is derived
from the *mapped* train interactions; the unmapped ones are still kept so the
sequence/ability baselines see the full history. To keep per-learner engine +
DKT runs tractable on 2.6 M instances, ``load_track`` subsamples learners
deterministically (seeded) — the cap is explicit and logged, never silent.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..events import Event
from ..adapters.slam import iter_exercises, parse_key
from ..adapters.slam_mapping import map_exercise

_STAMP = "slam.20190204"


@dataclass(frozen=True)
class Interaction:
    """One graded token: when, which concepts, and whether it was correct."""
    instance_id: str
    day: float
    node_ids: tuple[str, ...]   # mapped concept nodes ([] if the token maps to none)
    label: int                  # SLAM label: 1 = mistake, 0 = correct
    pos: str = ""               # token POS (for breakdowns)
    fmt: str = ""               # exercise format (reverse_translate / listen / ...)

    @property
    def correct(self) -> bool:
        return self.label == 0


# Train and eval items share the same shape.
EvalInstance = Interaction


@dataclass
class LearnerData:
    """One learner's history (``train``) and held-out items (``evalset``)."""
    user: str
    train: list[Interaction] = field(default_factory=list)
    evalset: list[Interaction] = field(default_factory=list)

    def train_events(self, *, source: str = "review") -> list[Event]:
        """Engine ``Event`` stream from the *mapped* train interactions."""
        return [
            Event(learner_id=self.user, node_ids=it.node_ids,
                  correct=it.correct, ts=it.day, source=source)
            for it in self.train if it.node_ids
        ]

    @property
    def eval_ref_day(self) -> float:
        """Causal reference time for prediction: the learner's first eval day.

        All train evidence is at or before this; mastery is read here so no
        future information leaks into a prediction.
        """
        return min((it.day for it in self.evalset), default=0.0)


def _interactions(path: str | Path, key: dict[str, int] | None, users: set[str] | None,
                  ) -> dict[str, list[Interaction]]:
    """Stream a SLAM file into ``{user: [Interaction, ...]}`` (optionally filtered)."""
    out: dict[str, list[Interaction]] = {}
    for ex in iter_exercises(path, key=key):
        if users is not None and ex.user not in users:
            continue
        node_map = map_exercise(ex)
        bucket = out.setdefault(ex.user, [])
        for tok in ex.tokens:
            if tok.label is None:
                continue
            bucket.append(Interaction(
                instance_id=tok.instance_id, day=ex.days,
                node_ids=node_map.get(tok.instance_id, ()), label=tok.label,
                pos=tok.pos, fmt=ex.format,
            ))
    return out


def _eval_users(path: str | Path) -> list[str]:
    """Distinct learners present in an eval file (sorted, for deterministic sampling)."""
    seen: set[str] = set()
    for ex in iter_exercises(path):
        seen.add(ex.user)
    return sorted(seen)


def load_track(
    data_dir: str | Path,
    *,
    course: str = "en_es",
    split: str = "dev",
    max_learners: int | None = None,
    seed: int = 0,
    min_train: int = 5,
    min_eval: int = 1,
) -> list[LearnerData]:
    """Load ``course`` (default en_es) into per-learner data for the eval harness.

    ``split`` selects the held-out file (``dev`` or ``test``); its ``.key`` labels
    are attached. ``max_learners`` caps the learner count (seeded sample of
    learners that appear in the eval split); ``None`` uses all of them. Learners
    with fewer than ``min_train`` train interactions or ``min_eval`` eval items
    are dropped. Each learner's interactions are returned time-sorted.
    """
    data_dir = Path(data_dir)
    train_path = data_dir / f"{course}.{_STAMP}.train"
    eval_path = data_dir / f"{course}.{_STAMP}.{split}"
    key_path = data_dir / f"{course}.{_STAMP}.{split}.key"
    for p in (train_path, eval_path, key_path):
        if not p.exists():
            raise FileNotFoundError(f"missing SLAM file: {p}")

    target: set[str] | None = None
    if max_learners is not None:
        users = _eval_users(eval_path)
        rng = random.Random(seed)
        target = set(rng.sample(users, min(max_learners, len(users))))

    key = parse_key(key_path)
    train = _interactions(train_path, key=None, users=target)
    evalset = _interactions(eval_path, key=key, users=target)

    learners: list[LearnerData] = []
    for user, evs in evalset.items():
        tr = train.get(user, [])
        if len(tr) < min_train or len(evs) < min_eval:
            continue
        tr.sort(key=lambda it: (it.day, it.instance_id))
        evs.sort(key=lambda it: (it.day, it.instance_id))
        learners.append(LearnerData(user=user, train=tr, evalset=evs))
    learners.sort(key=lambda ld: ld.user)
    return learners


def all_eval_instances(learners: Iterable[LearnerData]) -> list[Interaction]:
    """Flatten every learner's eval items (same order predictors emit scores)."""
    return [it for ld in learners for it in ld.evalset]


def cold_eval_mask(learners: Iterable[LearnerData]) -> list[bool]:
    """Per eval instance: True if the learner never practiced its node(s) in train.

    These "cold" nodes carry no direct evidence for that learner, so only the
    prerequisite graph can say anything about them — the slice where GNN
    propagation should help (the sharp RQ3 test). Aligned to
    ``all_eval_instances`` order. Unmapped instances are never cold (no node).
    """
    out: list[bool] = []
    for ld in learners:
        practiced: set[str] = set()
        for it in ld.train:
            practiced.update(it.node_ids)
        for it in ld.evalset:
            out.append(bool(it.node_ids) and all(n not in practiced for n in it.node_ids))
    return out
