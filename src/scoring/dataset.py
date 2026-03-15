"""PyTorch Dataset for 3D pose sequences with competition scores.

Scans output/poses_3d/ for Snowboard Men_* directories, parses the score
from the directory name, and loads poses_3d.json → tensor (T, 17, 3).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.scoring.graph import MIRROR_PAIRS, NUM_JOINTS


def _parse_score(stem: str) -> float | None:
    """Parse score from directory stem like 'Snowboard Men_1_80_Fabian Knözinger'."""
    # Pattern: Category_rank_score_Name
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return float(parts[2])
    except ValueError:
        return None


def load_poses(poses_3d_path: Path) -> np.ndarray:
    """Load poses_3d.json and return array of shape (T, 17, 3)."""
    with open(poses_3d_path) as f:
        data = json.load(f)
    frames = data["frames"]
    T = len(frames)
    poses = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)
    for t, frame in enumerate(frames):
        kpts = frame["keypoints_3d"]
        for j, xyz in enumerate(kpts):
            poses[t, j, 0] = xyz[0]
            poses[t, j, 1] = xyz[1]
            poses[t, j, 2] = xyz[2]
    return poses


def scan_snowboard_men(poses_dir: str | Path) -> list[tuple[str, float, Path]]:
    """Scan poses_3d directory and return (stem, score, path) for all Snowboard Men.

    Returns list sorted by score descending.
    """
    poses_dir = Path(poses_dir)
    results: list[tuple[str, float, Path]] = []
    for d in poses_dir.iterdir():
        if not d.is_dir():
            continue
        stem = d.name
        if not stem.startswith("Snowboard Men"):
            continue
        score = _parse_score(stem)
        if score is None or score == 0.0:
            continue
        poses_path = d / "poses_3d.json"
        if not poses_path.exists():
            continue
        results.append((stem, score, poses_path))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


if _TORCH_AVAILABLE:
    class PoseDataset(Dataset):
        """Variable-length pose sequence dataset with optional augmentation.

        Parameters
        ----------
        records : list[tuple[str, float, Path]]
            List of (stem, score, poses_3d_path).
        crop_len : int | None
            If set, randomly crop sequences to this length during __getitem__.
            Set to None to return full sequences (for inference).
        augment : bool
            If True, apply left-right mirror and Gaussian noise augmentations.
        noise_std : float
            Std of Gaussian noise added to coordinates when augment=True.
        """

        def __init__(
            self,
            records: list[tuple[str, float, Path]],
            *,
            crop_len: int | None = 512,
            augment: bool = False,
            noise_std: float = 0.005,
        ) -> None:
            self.records = records
            self.crop_len = crop_len
            self.augment = augment
            self.noise_std = noise_std
            # Pre-load all poses into memory
            self._poses: list[np.ndarray] = []
            self._scores: list[float] = []
            self._stems: list[str] = []
            for stem, score, path in records:
                self._poses.append(load_poses(path))
                self._scores.append(score)
                self._stems.append(stem)

        @property
        def stems(self) -> list[str]:
            return self._stems

        @property
        def scores(self) -> list[float]:
            return self._scores

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            poses = self._poses[idx].copy()  # (T, 17, 3)
            score = self._scores[idx]

            # Temporal random crop
            T = poses.shape[0]
            if self.crop_len is not None and T > self.crop_len:
                start = np.random.randint(0, T - self.crop_len + 1)
                poses = poses[start : start + self.crop_len]
            elif self.crop_len is not None and T < self.crop_len:
                # Pad with last frame
                pad = np.repeat(poses[-1:], self.crop_len - T, axis=0)
                poses = np.concatenate([poses, pad], axis=0)

            # Augmentations
            if self.augment:
                # Left-right mirror with 50% probability
                if np.random.rand() < 0.5:
                    poses = _mirror_lr(poses)
                # Gaussian coordinate noise
                poses = poses + np.random.randn(*poses.shape).astype(np.float32) * self.noise_std

            # Convert to (3, T, 17) for model input
            poses_t = torch.from_numpy(poses).float()  # (T, 17, 3)
            poses_t = poses_t.permute(2, 0, 1)  # (3, T, 17)

            return poses_t, torch.tensor(score, dtype=torch.float32)

        def get_full_sequence(self, idx: int) -> tuple[torch.Tensor, float, str]:
            """Return full-length sequence without cropping, for inference."""
            poses = self._poses[idx]  # (T, 17, 3)
            poses_t = torch.from_numpy(poses).float()
            poses_t = poses_t.permute(2, 0, 1)  # (3, T, 17)
            return poses_t, self._scores[idx], self._stems[idx]


def _mirror_lr(poses: np.ndarray) -> np.ndarray:
    """Swap left/right joints and flip X coordinate."""
    poses = poses.copy()
    for l_idx, r_idx in MIRROR_PAIRS:
        poses[:, [l_idx, r_idx]] = poses[:, [r_idx, l_idx]]
    # Flip X axis so mirrored pose is consistent
    poses[:, :, 0] = -poses[:, :, 0]
    return poses
