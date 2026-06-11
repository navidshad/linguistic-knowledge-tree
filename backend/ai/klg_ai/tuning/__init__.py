"""Tuning layer — personalization and weight learning over the core engine.

Closed-form per-learner edge tuning (``kgt``) and gradient weight-fitting
(``retrain`` — the differentiable propagation forward + per-learner fit, reused
by the global trainer and the live retrain endpoint). Depends only on
``klg_ai.core``.
"""
