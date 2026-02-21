"""3D pose lifting from 2D keypoint sequences.

Uses MotionBERT (Zhu et al., 2023) to lift 2D poses into 3D.
MotionBERT uses a temporal transformer that ingests a window of 2D poses
and outputs 3D coordinates in camera-relative space.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_2d_poses(poses_2d_path: str | Path) -> np.ndarray:
    """Load 2D poses JSON into an (N, 17, 3) array [x, y, confidence]."""
    with open(poses_2d_path) as f:
        data = json.load(f)
    frames = data["frames"]
    keypoints = np.array([f["keypoints"] for f in frames], dtype=np.float32)
    return keypoints  # (N, 17, 3)


def lift_to_3d(
    poses_2d_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "motionbert",
    device: str = "mps",
    receptive_field: int = 243,
) -> Path:
    """Lift 2D poses to 3D using a temporal model.

    Parameters
    ----------
    poses_2d_path : path to the poses_2d.json produced by the 2D stage.
    output_dir : directory for output files.
    model_name : lifting model to use.
    device : torch device string.
    receptive_field : temporal window for MotionBERT.

    Returns
    -------
    Path to the output poses_3d.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kpts_2d = load_2d_poses(poses_2d_path)  # (N, 17, 3)
    n_frames = kpts_2d.shape[0]

    print(f"Lifting {n_frames} frames to 3D  (model={model_name}, window={receptive_field})")

    if model_name == "motionbert":
        poses_3d = _lift_motionbert(kpts_2d, device=device, receptive_field=receptive_field)
    else:
        print(f"⚠  Unknown model '{model_name}' — using a naive baseline lift.")
        poses_3d = _lift_naive(kpts_2d)

    # Save results
    out_path = output_dir / "poses_3d.json"
    result = {
        "model": model_name,
        "n_frames": int(poses_3d.shape[0]),
        "keypoint_names": _load_keypoint_names(poses_2d_path),
        "frames": [
            {
                "frame_id": i,
                "keypoints_3d": poses_3d[i].tolist(),  # (17, 3)
            }
            for i in range(poses_3d.shape[0])
        ],
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  → 3D poses saved to {out_path}")
    return out_path


def _load_keypoint_names(poses_2d_path: str | Path) -> list[str]:
    with open(poses_2d_path) as f:
        data = json.load(f)
    return data.get("keypoint_names", [])


def _lift_motionbert(
    kpts_2d: np.ndarray,
    *,
    device: str = "mps",
    receptive_field: int = 243,
) -> np.ndarray:
    """Lift 2D→3D using MotionBERT.

    This is a placeholder showing the integration pattern.
    Full implementation requires downloading the MotionBERT checkpoint
    and loading the DSTformer model.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for MotionBERT lifting") from exc

    n_frames, n_joints, _ = kpts_2d.shape

    # Normalise 2D keypoints to [-1, 1] range
    xy = kpts_2d[:, :, :2].copy()
    confidence = kpts_2d[:, :, 2:3]

    # Centre on hip midpoint (joints 11, 12 in COCO)
    hip_center = (xy[:, 11:12, :] + xy[:, 12:13, :]) / 2.0
    xy = xy - hip_center

    # Scale so max extent ≈ 1
    scale = np.abs(xy).max(axis=(1, 2), keepdims=True) + 1e-6
    xy = xy / scale

    # TODO: Load real MotionBERT checkpoint and run inference.
    #
    # The integration pattern is:
    #   from motionbert.model import DSTformer
    #   model = DSTformer(...)
    #   model.load_state_dict(torch.load(checkpoint))
    #   model.eval().to(device)
    #
    #   # Pad/window the input to `receptive_field` frames
    #   input_2d = torch.from_numpy(xy).float().unsqueeze(0).to(device)
    #   with torch.no_grad():
    #       output_3d = model(input_2d)
    #   poses_3d = output_3d.squeeze(0).cpu().numpy()

    print("  ⚠  Using naive depth estimation (MotionBERT checkpoint not loaded)")
    return _lift_naive(kpts_2d)


def _lift_naive(kpts_2d: np.ndarray) -> np.ndarray:
    """Baseline: estimate Z from bone lengths heuristic.

    This is a simple placeholder — it creates a plausible-looking 3D skeleton
    by assigning depth proportional to vertical position (lower = closer).
    It is NOT accurate but lets you test the full pipeline.
    """
    n_frames, n_joints, _ = kpts_2d.shape
    xy = kpts_2d[:, :, :2].copy()

    # Centre on hip midpoint
    hip_center = (xy[:, 11:12, :] + xy[:, 12:13, :]) / 2.0
    xy = xy - hip_center

    # Normalise
    scale = np.abs(xy).max(axis=(1, 2), keepdims=True) + 1e-6
    xy = xy / scale

    # Fake Z: use a simple heuristic
    z = np.zeros((n_frames, n_joints, 1), dtype=np.float32)
    z[:, :, 0] = -xy[:, :, 1] * 0.3  # higher points slightly further away

    poses_3d = np.concatenate([xy, z], axis=-1)  # (N, 17, 3)
    return poses_3d
