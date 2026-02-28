"""Unit tests for velocity-consistent candidate scoring.

Tests the _velocity_consistency() helper in isolation with synthetic data.
"""

from __future__ import annotations

import collections
import math

import pytest

from src.tracking.tracker import _velocity_consistency


# ---------------------------------------------------------------------------
# Cold start (insufficient history) → neutral score
# ---------------------------------------------------------------------------

class TestColdStart:
    """Velocity consistency returns neutral 0.5 when history is too short."""

    def test_empty_history(self) -> None:
        hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        assert _velocity_consistency(hist, 5.0, 3.0) == pytest.approx(0.5)

    def test_one_entry(self) -> None:
        hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        hist.append((4.0, 2.0))
        assert _velocity_consistency(hist, 5.0, 3.0) == pytest.approx(0.5)

    def test_two_entries(self) -> None:
        hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        hist.append((4.0, 2.0))
        hist.append((5.0, 2.5))
        assert _velocity_consistency(hist, 5.0, 3.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Same direction, same speed → high score
# ---------------------------------------------------------------------------

class TestSameDirection:
    """Candidate moving in the same direction as history → high score."""

    def test_identical_velocity(self) -> None:
        hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        for _ in range(5):
            hist.append((10.0, 5.0))
        score = _velocity_consistency(hist, 10.0, 5.0)
        assert score > 0.9, f"Expected >0.9, got {score}"

    def test_similar_velocity(self) -> None:
        hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        for _ in range(5):
            hist.append((10.0, 5.0))
        # Slightly different but same direction
        score = _velocity_consistency(hist, 11.0, 5.5)
        assert score > 0.8, f"Expected >0.8, got {score}"


# ---------------------------------------------------------------------------
# Opposite direction → low score
# ---------------------------------------------------------------------------

class TestOppositeDirection:
    """Candidate moving opposite to history → low score."""

    def test_opposite_velocity(self) -> None:
        hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        for _ in range(5):
            hist.append((10.0, 5.0))
        score = _velocity_consistency(hist, -10.0, -5.0)
        assert score < 0.4, f"Expected <0.4, got {score}"

    def test_perpendicular_velocity(self) -> None:
        hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        for _ in range(5):
            hist.append((10.0, 0.0))
        # Perpendicular direction
        score = _velocity_consistency(hist, 0.0, 10.0)
        # Should be moderate — direction is 90° off but magnitude matches
        assert score < 0.8, f"Expected <0.8, got {score}"


# ---------------------------------------------------------------------------
# Stationary candidate when history shows motion → low score
# ---------------------------------------------------------------------------

class TestStationaryCandidate:
    """Stationary candidate when athlete is moving → low score."""

    def test_stationary_vs_moving(self) -> None:
        hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        for _ in range(5):
            hist.append((15.0, 8.0))
        score = _velocity_consistency(hist, 0.0, 0.0)
        assert score < 0.3, f"Expected <0.3, got {score}"

    def test_slow_vs_fast(self) -> None:
        hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        for _ in range(5):
            hist.append((20.0, 10.0))
        # Same direction but much slower (bystander)
        score = _velocity_consistency(hist, 2.0, 1.0)
        # Direction is right but magnitude is way off → moderate-low
        assert score < 0.75, f"Expected <0.75, got {score}"
        # Should still be lower than matching velocity
        perfect = _velocity_consistency(hist, 20.0, 10.0)
        assert score < perfect, f"Slow candidate ({score}) should score < matching ({perfect})"


# ---------------------------------------------------------------------------
# Weighted recency — recent entries matter more
# ---------------------------------------------------------------------------

class TestRecencyWeighting:
    """More recent velocity entries should have higher weight."""

    def test_recent_entries_dominate(self) -> None:
        hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        # Old entries: moving right
        hist.append((10.0, 0.0))
        hist.append((10.0, 0.0))
        # Recent entries: clearly turning downward (majority)
        hist.append((0.0, 15.0))
        hist.append((0.0, 20.0))
        hist.append((0.0, 22.0))

        # Candidate continues the recent trend (downward)
        score_trend = _velocity_consistency(hist, 0.0, 25.0)
        # Candidate continues old direction (rightward)
        score_old = _velocity_consistency(hist, 10.0, 0.0)

        assert score_trend > score_old, (
            f"Recent-trend candidate ({score_trend}) should score higher "
            f"than old-direction candidate ({score_old})"
        )


# ---------------------------------------------------------------------------
# Score always in [0, 1]
# ---------------------------------------------------------------------------

class TestScoreRange:
    """Score must always be in [0, 1] range regardless of inputs."""

    @pytest.mark.parametrize("vx,vy", [
        (0.0, 0.0),
        (100.0, 100.0),
        (-50.0, 50.0),
        (0.01, -0.01),
        (1000.0, 0.0),
    ])
    def test_score_bounded(self, vx: float, vy: float) -> None:
        hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        for _ in range(5):
            hist.append((10.0, -5.0))
        score = _velocity_consistency(hist, vx, vy)
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"
