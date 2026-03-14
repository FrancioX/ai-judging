"""Utilities for converting 2D keypoints to MotionBERT's H36M-17 convention."""

from __future__ import annotations

import numpy as np


COCO_KEYPOINTS_17 = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# H36M-17 joint order used by common MotionBERT evaluation tooling.
H36M_KEYPOINTS_17 = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

# Bone links for H36M-17 plotting.
H36M_SKELETON_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]


def convert_keypoints_to_h36m_17(
    keypoints: np.ndarray,
    keypoint_names: list[str],
) -> tuple[np.ndarray, list[str], bool]:
    """Convert keypoints to H36M-17 order expected by MotionBERT.

    Parameters
    ----------
    keypoints :
        Keypoint tensor with shape (N, 17, 3) in [x, y, confidence] format.
    keypoint_names :
        Names for the 17 joints that define keypoint ordering.

    Returns
    -------
    tuple[np.ndarray, list[str], bool]
        (converted_keypoints, output_keypoint_names, did_convert)
    """
    _validate_shape(keypoints)
    normalized = [name.lower() for name in keypoint_names]
    coco_normalized = [name.lower() for name in COCO_KEYPOINTS_17]
    h36m_normalized = [name.lower() for name in H36M_KEYPOINTS_17]

    if normalized == h36m_normalized:
        return keypoints, H36M_KEYPOINTS_17, False

    if normalized != coco_normalized:
        raise RuntimeError(
            "Unsupported keypoint convention for MotionBERT conversion. "
            "Expected COCO-17 or H36M-17 keypoint_names."
        )

    converted = _convert_coco_to_h36m(keypoints)
    return converted, H36M_KEYPOINTS_17, True


def get_hip_indices(keypoint_names: list[str]) -> tuple[int, int]:
    """Return (left_hip_index, right_hip_index) for a keypoint-name list."""
    if not keypoint_names:
        raise RuntimeError("keypoint_names is empty; cannot infer hip indices.")

    lowered = [name.lower() for name in keypoint_names]
    try:
        left = lowered.index("left_hip")
        right = lowered.index("right_hip")
    except ValueError as exc:
        raise RuntimeError(
            "keypoint_names must include left_hip and right_hip for centering."
        ) from exc
    return left, right


def _validate_shape(keypoints: np.ndarray) -> None:
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise RuntimeError(
            "Expected keypoints with shape (N, 17, 3), "
            f"got {tuple(keypoints.shape)}."
        )


def _convert_coco_to_h36m(coco: np.ndarray) -> np.ndarray:
    idx = {name: i for i, name in enumerate(COCO_KEYPOINTS_17)}
    out = np.zeros_like(coco, dtype=np.float32)

    left_hip = coco[:, idx["left_hip"], :]
    right_hip = coco[:, idx["right_hip"], :]
    left_shoulder = coco[:, idx["left_shoulder"], :]
    right_shoulder = coco[:, idx["right_shoulder"], :]

    pelvis = _midpoint(left_hip, right_hip)
    thorax = _midpoint(left_shoulder, right_shoulder)
    spine = _midpoint(pelvis, thorax)
    neck = thorax.copy()
    head = _estimate_head(coco, thorax)

    out[:, 0, :] = pelvis
    out[:, 1, :] = right_hip
    out[:, 2, :] = coco[:, idx["right_knee"], :]
    out[:, 3, :] = coco[:, idx["right_ankle"], :]
    out[:, 4, :] = left_hip
    out[:, 5, :] = coco[:, idx["left_knee"], :]
    out[:, 6, :] = coco[:, idx["left_ankle"], :]
    out[:, 7, :] = spine
    out[:, 8, :] = thorax
    out[:, 9, :] = neck
    out[:, 10, :] = head
    out[:, 11, :] = left_shoulder
    out[:, 12, :] = coco[:, idx["left_elbow"], :]
    out[:, 13, :] = coco[:, idx["left_wrist"], :]
    out[:, 14, :] = right_shoulder
    out[:, 15, :] = coco[:, idx["right_elbow"], :]
    out[:, 16, :] = coco[:, idx["right_wrist"], :]

    return out


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xy = (a[:, :2] + b[:, :2]) / 2.0
    conf = np.minimum(a[:, 2], b[:, 2])[:, None]
    return np.concatenate([xy, conf], axis=1)


def _estimate_head(coco: np.ndarray, thorax: np.ndarray) -> np.ndarray:
    idx = {name: i for i, name in enumerate(COCO_KEYPOINTS_17)}
    face = np.stack(
        [
            coco[:, idx["nose"], :],
            coco[:, idx["left_eye"], :],
            coco[:, idx["right_eye"], :],
            coco[:, idx["left_ear"], :],
            coco[:, idx["right_ear"], :],
        ],
        axis=1,
    )

    conf = face[:, :, 2]
    weight_sum = np.sum(conf, axis=1, keepdims=True)
    weighted_xy = np.sum(face[:, :, :2] * conf[:, :, None], axis=1)

    head = thorax.copy()
    valid = (weight_sum[:, 0] > 0.0)
    head[valid, :2] = weighted_xy[valid] / np.maximum(weight_sum[valid], 1e-6)
    head[valid, 2] = np.max(conf[valid], axis=1)
    return head
