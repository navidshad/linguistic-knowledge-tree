"""Adapters: map evidence sources to the engine's `Event` schema.

`synthetic` generates deterministic learner event streams (the demo learner and
parametric test/validation learners). Real adapters land here later: a Duolingo
SLAM / EdNet adapter for thesis validation (Phase 5), then a Subturtle adapter
(`leitner_review_log` + phrase/chunk/dialog evidence) for productization.
"""
from klg_ai.data.adapters.synthetic import DEMO_KNOWN, demo_events, generate_events

__all__ = ["DEMO_KNOWN", "demo_events", "generate_events"]
