"""Tests for 3D lifting and 3D side-by-side visualization."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pose_3d.lifter import lift_to_3d
from src.visualization.render import visualize_3d_side_by_side


def _write_dummy_poses_2d(path: Path, n_frames: int = 5) -> None:
    keypoint_names = [f"kp_{i}" for i in range(17)]
    frames = []
    for frame_id in range(n_frames):
        keypoints = []
        for joint_id in range(17):
            keypoints.append([100.0 + joint_id, 200.0 + frame_id, 0.9])
        frames.append({"frame_id": frame_id, "keypoints": keypoints})

    with open(path, "w") as f:
        json.dump({"keypoint_names": keypoint_names, "frames": frames}, f)


def _write_dummy_frames(frame_dir: Path, n_frames: int = 5) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(n_frames):
        img = np.full((240, 320, 3), 200, dtype=np.uint8)
        cv2.putText(img, f"f{idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(str(frame_dir / f"frame_{idx:06d}.jpg"), img)


def _write_dummy_poses_3d(path: Path, n_frames: int = 5) -> None:
    keypoint_names = [f"kp_{i}" for i in range(17)]
    frames = []
    for frame_id in range(n_frames):
        joints = []
        for joint_id in range(17):
            joints.append([joint_id * 0.05, frame_id * 0.02, (joint_id - 8) * 0.03])
        frames.append({"frame_id": frame_id, "keypoints_3d": joints})

    with open(path, "w") as f:
        json.dump({"keypoint_names": keypoint_names, "frames": frames}, f)


def test_lift_to_3d_requires_motionbert_checkpoint(tmp_path: Path) -> None:
    poses_2d_path = tmp_path / "poses_2d.json"
    out_dir = tmp_path / "poses_3d"
    _write_dummy_poses_2d(poses_2d_path)

    with pytest.raises(FileNotFoundError, match="No MotionBERT checkpoint configured"):
        lift_to_3d(
            poses_2d_path,
            out_dir,
            model_name="motionbert",
            checkpoint_path=None,
            checkpoint_url=None,
        )


def test_visualize_3d_side_by_side_writes_mp4(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    poses_3d_path = tmp_path / "poses_3d.json"
    output_dir = tmp_path / "viz"

    _write_dummy_frames(frame_dir, n_frames=6)
    _write_dummy_poses_3d(poses_3d_path, n_frames=6)

    out_path = visualize_3d_side_by_side(frame_dir, poses_3d_path, output_dir, fps=10.0)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
