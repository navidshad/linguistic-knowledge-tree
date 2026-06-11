"""Forgetting: exponential recency decay (thesis RQ4 building block)."""
from klg_ai.core.forgetting import recency_weight


def test_no_age_is_full_weight():
    assert recency_weight(0.0, 30.0) == 1.0


def test_one_half_life_halves():
    assert recency_weight(30.0, 30.0) == 0.5


def test_two_half_lives_quarter():
    assert abs(recency_weight(60.0, 30.0) - 0.25) < 1e-9


def test_disabled_is_no_decay():
    assert recency_weight(365.0, 30.0, enabled=False) == 1.0


def test_nonpositive_half_life_is_no_decay():
    assert recency_weight(100.0, 0.0) == 1.0
