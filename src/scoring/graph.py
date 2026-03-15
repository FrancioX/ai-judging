"""H36M-17 skeleton graph definition for ST-GCN.

Defines the adjacency matrix with centripetal spatial partitioning
(self / root-ward / leaf-ward) as in the original ST-GCN paper.
"""

from __future__ import annotations

import numpy as np

# Number of joints in H36M-17
NUM_JOINTS = 17

# H36M-17 joint indices
# pelvis(0) → right_hip(1) → right_knee(2) → right_ankle(3)
# pelvis(0) → left_hip(4)  → left_knee(5)  → left_ankle(6)
# pelvis(0) → spine(7) → thorax(8) → neck(9) → head(10)
# thorax(8) → left_shoulder(11)  → left_elbow(12)  → left_wrist(13)
# thorax(8) → right_shoulder(14) → right_elbow(15) → right_wrist(16)

EDGES: list[tuple[int, int]] = [
    # Right leg
    (0, 1), (1, 2), (2, 3),
    # Left leg
    (0, 4), (4, 5), (5, 6),
    # Spine / head
    (0, 7), (7, 8), (8, 9), (9, 10),
    # Left arm
    (8, 11), (11, 12), (12, 13),
    # Right arm
    (8, 14), (14, 15), (15, 16),
]

# Left-right mirror joint pairs for data augmentation
MIRROR_PAIRS: list[tuple[int, int]] = [
    (1, 4),   # right_hip  ↔ left_hip
    (2, 5),   # right_knee ↔ left_knee
    (3, 6),   # right_ankle ↔ left_ankle
    (11, 14), # left_shoulder ↔ right_shoulder
    (12, 15), # left_elbow ↔ right_elbow
    (13, 16), # left_wrist ↔ right_wrist
]


def _normalize(A: np.ndarray) -> np.ndarray:
    """Symmetric D^{-1/2} A D^{-1/2} normalization row by row."""
    rowsum = A.sum(axis=1)
    d_inv_sqrt = np.where(rowsum > 0, 1.0 / np.sqrt(np.maximum(rowsum, 1e-12)), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def build_adjacency() -> np.ndarray:
    """Build normalized centripetal-partitioned adjacency matrix.

    Returns
    -------
    np.ndarray
        Shape (3, N, N) — [self, root-ward, leaf-ward].
        Each partition is D^{-1/2} A D^{-1/2} normalized.
    """
    N = NUM_JOINTS
    A_self = np.eye(N, dtype=np.float32)

    # root-ward: child contributes to parent  (A[parent, child] = 1)
    A_root = np.zeros((N, N), dtype=np.float32)
    # leaf-ward: parent contributes to child  (A[child, parent] = 1)
    A_leaf = np.zeros((N, N), dtype=np.float32)

    for parent, child in EDGES:
        A_root[parent, child] = 1.0  # centripetal: toward root
        A_leaf[child, parent] = 1.0  # centrifugal: away from root

    A_norm = np.stack([
        _normalize(A_self),
        _normalize(A_root),
        _normalize(A_leaf),
    ], axis=0)
    return A_norm
