"""Hand-crafted feature baseline: joint angles + kinematics → Ridge regression.

Feature vector per video (~144 dimensions):
  - 12 joint angles (per frame) × {mean, std, min, max, range, skewness} = 72
  - 17 velocity magnitudes (per frame) × {mean, std, max, range} = 68
  - body height ratio (per frame) × {mean, std, max, range} = 4
  → ~144 features total

Regressor: Ridge regression (sklearn), robust with small N.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.stats import skew as _scipy_skew
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ── Joint angle definitions (triplets: a → b → c, angle at b) ──────────────
# Each triplet is (joint_a, joint_b, joint_c) — angle measured at joint_b.
ANGLE_TRIPLETS: list[tuple[int, int, int]] = [
    (1, 2, 3),    # right knee: right_hip → right_knee → right_ankle
    (4, 5, 6),    # left knee:  left_hip  → left_knee  → left_ankle
    (0, 1, 2),    # right hip:  pelvis → right_hip → right_knee
    (0, 4, 5),    # left hip:   pelvis → left_hip  → left_knee
    (14, 15, 16), # right elbow: right_shoulder → right_elbow → right_wrist
    (11, 12, 13), # left elbow:  left_shoulder  → left_elbow  → left_wrist
    (8, 14, 15),  # right shoulder: thorax → right_shoulder → right_elbow
    (8, 11, 12),  # left shoulder:  thorax → left_shoulder  → left_elbow
    (0, 7, 8),    # spine bend: pelvis → spine → thorax
    (8, 9, 10),   # neck:       thorax → neck  → head
    (7, 8, 14),   # right trunk lean: spine → thorax → right_shoulder
    (7, 8, 11),   # left trunk lean:  spine → thorax → left_shoulder
]


def _angle_at_joint(poses: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Compute angle (radians) at joint b across T frames.

    Parameters
    ----------
    poses : np.ndarray  (T, 17, 3)
    a, b, c : int  joint indices

    Returns
    -------
    np.ndarray  (T,) angles in [0, π]
    """
    v1 = poses[:, a] - poses[:, b]  # (T, 3)
    v2 = poses[:, c] - poses[:, b]  # (T, 3)
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True).clip(1e-8)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True).clip(1e-8)
    cos_angle = (v1 / norm1 * v2 / norm2).sum(axis=1).clip(-1.0, 1.0)
    return np.arccos(cos_angle)  # (T,)


def _temporal_stats(signal: np.ndarray, include_skew: bool = True) -> np.ndarray:
    """Compute temporal statistics of a 1D signal.

    Returns [mean, std, min, max, range, skewness] if include_skew else [mean, std, max, range].
    """
    mean = signal.mean()
    std = signal.std()
    mn = signal.min()
    mx = signal.max()
    rng = mx - mn
    if include_skew:
        if _SCIPY:
            sk = float(_scipy_skew(signal))
        else:
            # Manual skewness
            n = len(signal)
            if n < 3 or std < 1e-10:
                sk = 0.0
            else:
                sk = float(((signal - mean) ** 3).mean() / (std ** 3))
        return np.array([mean, std, mn, mx, rng, sk], dtype=np.float32)
    return np.array([mean, std, mx, rng], dtype=np.float32)


def extract_features(poses: np.ndarray) -> np.ndarray:
    """Extract hand-crafted feature vector from pose sequence.

    Parameters
    ----------
    poses : np.ndarray  (T, 17, 3)

    Returns
    -------
    np.ndarray  (D,)  feature vector
    """
    feats: list[np.ndarray] = []

    # 1. Joint angles: 12 × 6 stats = 72
    for a, b, c in ANGLE_TRIPLETS:
        angles = _angle_at_joint(poses, a, b, c)  # (T,)
        feats.append(_temporal_stats(angles, include_skew=True))

    # 2. Joint velocity magnitudes: 17 × 4 stats = 68
    vel = np.diff(poses, axis=0)  # (T-1, 17, 3)
    vel_mag = np.linalg.norm(vel, axis=2)  # (T-1, 17)
    for j in range(17):
        feats.append(_temporal_stats(vel_mag[:, j], include_skew=False))

    # 3. Body height ratio (head-to-ankle) normalized by mean: 1 × 4 stats = 4
    #    head(10), left_ankle(6), right_ankle(3)
    ankle_mid = (poses[:, 3] + poses[:, 6]) / 2  # (T, 3)
    height = np.linalg.norm(poses[:, 10] - ankle_mid, axis=1)  # (T,)
    feats.append(_temporal_stats(height, include_skew=False))

    return np.concatenate(feats, axis=0)


class RidgeBaseline:
    """Ridge regression on hand-crafted features with z-score normalization."""

    def __init__(self, alpha: float = 1.0) -> None:
        if not _SKLEARN:
            raise ImportError("scikit-learn is required for RidgeBaseline")
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: (N, D), y: (N,)"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) → (N,)"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def fit_predict_loocv(
        self, features: list[np.ndarray], scores: list[float]
    ) -> np.ndarray:
        """Leave-one-out cross-validation.

        Parameters
        ----------
        features : list[np.ndarray]  N feature vectors
        scores : list[float]         N ground-truth scores

        Returns
        -------
        np.ndarray  (N,)  OOF predictions
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X = np.stack(features, axis=0)  # (N, D)
        y = np.array(scores, dtype=np.float32)
        N = len(scores)
        preds = np.zeros(N, dtype=np.float32)

        for i in range(N):
            mask = np.arange(N) != i
            X_tr, y_tr = X[mask], y[mask]
            X_te = X[i : i + 1]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = Ridge(alpha=self.alpha)
            model.fit(X_tr_s, y_tr)
            preds[i] = model.predict(X_te_s)[0]

        return preds
