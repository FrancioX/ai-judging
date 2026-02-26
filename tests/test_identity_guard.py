"""Unit tests for the identity guard — optical flow-based identity switch detection.

Tests the _maintain_of_trace() and _validate_identity() functions in isolation
with synthetic data.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tracking.tracker import _maintain_of_trace, _validate_identity


# ---------------------------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_dummy_frame(width: int, height: int, color: tuple[int, int, int] = (100, 100, 100)) -> np.ndarray:
    """Create a dummy BGR frame of specified size."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def create_moving_circle_frame(
    width: int, height: int, cx: int, cy: int, radius: int = 20,
) -> np.ndarray:
    """Create a frame with a white circle at the given center."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img, (cx, cy), radius, (255, 255, 255), -1)
    return img


def save_frame(frame: np.ndarray, frame_dir: Path, frame_id: int) -> str:
    """Save a frame and return its filename."""
    frame_file = f"frame_{frame_id:06d}.jpg"
    cv2.imwrite(str(frame_dir / frame_file), frame)
    return frame_file


@pytest.fixture
def dummy_seg_frames(temp_dir: Path) -> tuple[list[dict], Path]:
    """Create dummy segmentation frames with moving circle."""
    frame_dir = temp_dir / "frames"
    frame_dir.mkdir()

    n_frames = 100
    width, height = 640, 480
    frames = []

    # Create 100 frames with a circle moving slowly from left to right
    for fid in range(n_frames):
        cx = 200 + fid // 4  # Gradual horizontal movement
        cy = 240  # Center vertical
        img = create_moving_circle_frame(width, height, cx, cy)
        frame_file = save_frame(img, frame_dir, fid)
        frames.append({"frame_id": fid, "frame_file": frame_file})

    return frames, frame_dir


# ---------------------------------------------------------------------------
# Tests for _maintain_of_trace()
# ---------------------------------------------------------------------------

def test_maintain_of_trace_basic(dummy_seg_frames):
    """Test basic functionality: OF trace should track a moving circle."""
    seg_frames, frame_dir = dummy_seg_frames
    n_frames = len(seg_frames)
    width, height = 640, 480

    # Create synthetic selected_obs with detections every 10 frames
    selected_obs = {}
    for fid in range(0, n_frames, 10):
        cx = 200 + fid // 4
        cy = 240
        pad = 30

        selected_obs[fid] = {
            "bbox": [
                int(cx - pad),
                int(cy - pad),
                int(cx + pad),
                int(cy + pad),
            ],
            "confidence": 0.95,
            "track_id": 1,
        }

    # Run OF trace
    of_trace = _maintain_of_trace(
        selected_obs, n_frames, width, height, frame_dir, seg_frames,
        method="dense",  # Use dense for synthetic data
        reanchor_interval=5,
        reanchor_min_conf=0.5,
        max_drift_px=500.0,
    )

    # Verify OF trace is not empty
    assert len(of_trace) > 0, "OF trace should not be empty"

    # Verify OF trace has entries for key frames
    assert 0 in of_trace, "First frame should be in OF trace "
    assert n_frames - 1 in of_trace, "Last frame should be in OF trace"

    # The OF trace should show generally increasing x coordinates (rightward motion)
    first_cx = of_trace[0][0]
    last_cx = of_trace[n_frames - 1][0]
    assert last_cx > first_cx, "Circle should move right over time"


def test_maintain_of_trace_reanchoring(dummy_seg_frames):
    """Test that re-anchoring resets drift."""
    seg_frames, frame_dir = dummy_seg_frames
    n_frames = len(seg_frames)
    width, height = 640, 480

    # Create detections spaced out with high confidence
    selected_obs = {}
    for fid in range(0, n_frames, 5):
        cx = 200 + fid // 4
        cy = 240
        pad = 30

        selected_obs[fid] = {
            "bbox": [
                int(cx - pad),
                int(cy - pad),
                int(cx + pad),
                int(cy + pad),
            ],
            "confidence": 0.95,
            "track_id": 1,
        }

    # Run with small re-anchor interval
    of_trace = _maintain_of_trace(
        selected_obs, n_frames, width, height, frame_dir, seg_frames,
        method="dense",
        reanchor_interval=2,  # Re-anchor every 2 detections
        reanchor_min_conf=0.5,
        max_drift_px=100.0,
    )

    # Verify OF trace exists
    assert len(of_trace) > 0


def test_validate_identity_no_rejections(dummy_seg_frames):
    """Test validate_identity with spatially coherent detections."""
    seg_frames, frame_dir = dummy_seg_frames
    n_frames = len(seg_frames)
    width, height = 640, 480

    # Create OF trace showing steady motion
    of_trace = {}
    for fid in range(n_frames):
        of_cx = 200.0 + fid // 4
        of_cy = 240.0
        of_trace[fid] = (of_cx, of_cy)

    # Create detections that follow the OF trace closely
    selected_obs = {}
    for fid in range(0, n_frames, 5):
        cx = 200 + fid // 4  # Matches OF trace
        cy = 240
        pad = 30

        selected_obs[fid] = {
            "bbox": [
                int(cx - pad),
                int(cy - pad),
                int(cx + pad),
                int(cy + pad),
            ],
            "confidence": 0.95,
            "track_id": 1,
        }

    # Validate
    validated_obs, metadata = _validate_identity(
        selected_obs, of_trace,
        max_jump_px=150.0,
    )

    # No detections should be rejected (all close to OF trace)
    assert metadata["rejections"] == 0, "No detections should be rejected"
    assert len(validated_obs) == len(selected_obs), "No detections should be removed"


def test_validate_identity_reject_jump(dummy_seg_frames):
    """Test validate_identity rejects large jumps away from OF trace."""
    seg_frames, frame_dir = dummy_seg_frames
    n_frames = len(seg_frames)
    width, height = 640, 480

    # Create OF trace showing steady motion
    of_trace = {}
    for fid in range(n_frames):
        of_cx = 200.0 + fid // 4
        of_cy = 240.0
        of_trace[fid] = (of_cx, of_cy)

    # Create detections with a giant jump at frame 50
    selected_obs = {}
    for fid in range(0, n_frames, 5):
        if fid == 50:
            # Giant jump to the right (500px away from OF prediction)
            cx = 200 + fid // 4 + 500
            cy = 240
        else:
            cx = 200 + fid // 4  # Matches OF trace
            cy = 240
        pad = 30

        selected_obs[fid] = {
            "bbox": [
                int(cx - pad),
                int(cy - pad),
                int(cx + pad),
                int(cy + pad),
            ],
            "confidence": 0.95,
            "track_id": 1,
        }

    # Validate with moderate threshold
    validated_obs, metadata = _validate_identity(
        selected_obs, of_trace,
        max_jump_px=150.0,
    )

    # The frame at 50 should be rejected
    assert metadata["rejections"] > 0, "Jump should trigger rejection"
    assert 50 not in validated_obs, "Frame 50 should be removed"


def test_validate_identity_legitimate_motion():
    """Test validate_identity doesn't reject legitimate large motion (pan / zoomout)."""
    n_frames = 100
    width, height = 640, 480

    # Simulated large camera pan: both OF and detection agree on ~200px rightward shift
    of_trace = {}
    selected_obs = {}

    for fid in range(n_frames):
        # OF predicts smooth rightward motion
        of_cx = 200.0 + fid * 2  # 200px pan over 100 frames
        of_cy = 240.0
        of_trace[fid] = (of_cx, of_cy)

        if fid % 10 == 0:
            # Detections also show rightward motion
            cx = 200 + fid * 2
            cy = 240
            pad = 30

            selected_obs[fid] = {
                "bbox": [
                    int(cx - pad),
                    int(cy - pad),
                    int(cx + pad),
                    int(cy + pad),
                ],
                "confidence": 0.95,
                "track_id": 1,
            }

    # Validate with large threshold
    validated_obs, metadata = _validate_identity(
        selected_obs, of_trace,
        max_jump_px=150.0,
    )

    # No rejections because both detection and OF agree
    assert metadata["rejections"] == 0, "Consensual large motion should not be rejected"


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

def test_of_trace_and_validate_identity_integration(dummy_seg_frames):
    """Test full pipeline: maintain OF trace then validate identity."""
    seg_frames, frame_dir = dummy_seg_frames
    n_frames = len(seg_frames)
    width, height = 640, 480

    # Create detections with good coverage
    selected_obs = {}
    for fid in range(0, n_frames, 10):
        cx = 200 + fid // 4
        cy = 240
        pad = 30

        selected_obs[fid] = {
            "bbox": [
                int(cx - pad),
                int(cy - pad),
                int(cx + pad),
                int(cy + pad),
            ],
            "confidence": 0.95,
            "track_id": 1,
        }

    # Build OF trace
    of_trace = _maintain_of_trace(
        selected_obs, n_frames, width, height, frame_dir, seg_frames,
        method="dense",
        reanchor_interval=5,
        reanchor_min_conf=0.5,
        max_drift_px=500.0,
    )

    # Now introduce a bad detection
    selected_obs_with_jump = dict(selected_obs)
    selected_obs_with_jump[50] = {
        "bbox": [500, 200, 560, 280],  # Far from OF prediction
        "confidence": 0.95,
        "track_id": 1,
    }

    # Validate
    validated_obs, metadata = _validate_identity(
        selected_obs_with_jump, of_trace,
        max_jump_px=150.0,
    )

    # The bad detection at frame 50 should be filtered out
    assert metadata["rejections"] > 0
    assert 50 not in validated_obs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
