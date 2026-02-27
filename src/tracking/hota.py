"""HOTA (Higher Order Tracking Accuracy) metric for single-object tracking evaluation.

HOTA balances detection quality (DetA) and association quality (AssA) to measure
how well a tracker maintains the correct identity of an athlete over time.

This is a simplified single-object implementation adapted for freeride skiing
judging, where we only track one athlete per video and have center-point
annotations rather than full bounding boxes.

Reference: Luiten et al., "HOTA: A Higher Order Metric for Evaluating
Multi-Object Tracking" (IJCV 2021)
"""

from __future__ import annotations

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def compute_hota(
    predictions: dict[int, tuple[float, float]],
    ground_truth: dict[int, tuple[float, float]],
    distance_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Compute HOTA metric for single-object tracking.

    HOTA (Higher Order Tracking Accuracy) is computed as the geometric mean
    of DetA (detection accuracy) and AssA (association accuracy) across
    multiple distance thresholds.

    Parameters
    ----------
    predictions : dict mapping frame_id → (center_x, center_y)
        Predicted track centers for each frame.
    ground_truth : dict mapping frame_id → (center_x, center_y)
        Ground truth centers for each frame (interpolated from keyframes).
    distance_thresholds : list of distance thresholds in pixels.
        Defaults to [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] pixels.
        These approximate IoU thresholds [0.5, 0.6, ..., 0.95] for typical
        skier bounding boxes (~100x200px).

    Returns
    -------
    dict with keys:
        - "hota": HOTA score (harmonic mean of DetA and AssA)
        - "det_a": Detection accuracy
        - "ass_a": Association accuracy
        - "num_frames": Number of frames evaluated
        - "num_thresholds": Number of distance thresholds used
    """
    if not _HAS_NUMPY:
        raise ImportError("NumPy is required for HOTA computation")

    if distance_thresholds is None:
        # Default: 10px steps from 10 to 100px
        # Approximate IoU thresholds from 0.5 to 0.95 for typical skier bbox
        distance_thresholds = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    # Get common frame IDs (frames present in both predictions and GT)
    common_frames = sorted(set(predictions.keys()) & set(ground_truth.keys()))

    if not common_frames:
        # No overlap between predictions and ground truth
        return {
            "hota": 0.0,
            "det_a": 0.0,
            "ass_a": 0.0,
            "num_frames": 0,
            "num_thresholds": len(distance_thresholds),
        }

    # Compute DetA and AssA for each distance threshold
    det_a_scores = []
    ass_a_scores = []

    for threshold in distance_thresholds:
        det_a = _compute_detection_accuracy(
            predictions, ground_truth, common_frames, threshold
        )
        ass_a = _compute_association_accuracy(
            predictions, ground_truth, common_frames, threshold
        )
        det_a_scores.append(det_a)
        ass_a_scores.append(ass_a)

    # HOTA is the geometric mean of DetA and AssA, averaged across thresholds
    # HOTA(α) = √(DetA(α) × AssA(α)) for each threshold α
    # Final HOTA = mean over all thresholds
    hota_per_threshold = np.sqrt(np.array(det_a_scores) * np.array(ass_a_scores))
    hota = float(np.mean(hota_per_threshold))

    # Average DetA and AssA across thresholds for reporting
    det_a_avg = float(np.mean(det_a_scores))
    ass_a_avg = float(np.mean(ass_a_scores))

    return {
        "hota": hota,
        "det_a": det_a_avg,
        "ass_a": ass_a_avg,
        "num_frames": len(common_frames),
        "num_thresholds": len(distance_thresholds),
    }


def _compute_detection_accuracy(
    predictions: dict[int, tuple[float, float]],
    ground_truth: dict[int, tuple[float, float]],
    common_frames: list[int],
    distance_threshold: float,
) -> float:
    """Compute detection accuracy at a given distance threshold.

    DetA measures how well predictions match ground truth spatially.
    For single-object tracking, this is the fraction of frames where
    the predicted center is within distance_threshold of the GT center.

    Parameters
    ----------
    predictions : predicted centers per frame
    ground_truth : ground truth centers per frame
    common_frames : list of frame IDs to evaluate
    distance_threshold : distance threshold in pixels

    Returns
    -------
    Detection accuracy in [0, 1]
    """
    if not common_frames:
        return 0.0

    num_matches = 0

    for frame_id in common_frames:
        pred_x, pred_y = predictions[frame_id]
        gt_x, gt_y = ground_truth[frame_id]

        # Euclidean distance
        distance = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

        if distance <= distance_threshold:
            num_matches += 1

    # DetA = True Positives / (True Positives + False Positives + False Negatives)
    # For single-object tracking with 100% detection rate:
    # DetA = num_matches / total_frames
    det_a = num_matches / len(common_frames)

    return det_a


def _compute_association_accuracy(
    predictions: dict[int, tuple[float, float]],
    ground_truth: dict[int, tuple[float, float]],
    common_frames: list[int],
    distance_threshold: float,
) -> float:
    """Compute association accuracy at a given distance threshold.

    AssA measures identity consistency over time. For single-object tracking,
    this evaluates whether the track maintains the correct identity throughout.

    We use a temporal consistency metric: for each matched detection, check
    how well it aligns with the trajectory formed by other matched detections.

    Parameters
    ----------
    predictions : predicted centers per frame
    ground_truth : ground truth centers per frame
    common_frames : list of frame IDs to evaluate
    distance_threshold : distance threshold in pixels

    Returns
    -------
    Association accuracy in [0, 1]
    """
    if len(common_frames) < 2:
        # Need at least 2 frames to measure association
        return 1.0 if len(common_frames) == 1 else 0.0

    # First, identify which frames are "matched" (within threshold)
    matched_frames = []
    for frame_id in common_frames:
        pred_x, pred_y = predictions[frame_id]
        gt_x, gt_y = ground_truth[frame_id]
        distance = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

        if distance <= distance_threshold:
            matched_frames.append(frame_id)

    if len(matched_frames) < 2:
        # No consistent track to evaluate
        return 0.0

    # AssA: Measure consistency of matched detections
    # For single-object tracking, we check if matched detections form
    # a coherent trajectory by computing average pairwise consistency

    # Simple approach: ratio of matched frames to total frames
    # This rewards maintaining matches over time (identity preservation)
    # More sophisticated: compute trajectory prediction error

    # For single-object with full coverage, AssA ≈ fraction of consecutive matches
    consecutive_matches = 0
    total_pairs = len(common_frames) - 1

    for i in range(len(common_frames) - 1):
        frame_a = common_frames[i]
        frame_b = common_frames[i + 1]

        # Check if both frames are matched
        pred_a_x, pred_a_y = predictions[frame_a]
        gt_a_x, gt_a_y = ground_truth[frame_a]
        dist_a = np.sqrt((pred_a_x - gt_a_x) ** 2 + (pred_a_y - gt_a_y) ** 2)

        pred_b_x, pred_b_y = predictions[frame_b]
        gt_b_x, gt_b_y = ground_truth[frame_b]
        dist_b = np.sqrt((pred_b_x - gt_b_x) ** 2 + (pred_b_y - gt_b_y) ** 2)

        if dist_a <= distance_threshold and dist_b <= distance_threshold:
            # Both matched → check trajectory consistency
            # Predicted velocity vs GT velocity should be similar
            pred_vel = np.array([pred_b_x - pred_a_x, pred_b_y - pred_a_y])
            gt_vel = np.array([gt_b_x - gt_a_x, gt_b_y - gt_a_y])

            # Cosine similarity of velocity vectors (direction consistency)
            pred_vel_norm = np.linalg.norm(pred_vel)
            gt_vel_norm = np.linalg.norm(gt_vel)

            if pred_vel_norm > 0 and gt_vel_norm > 0:
                cosine_sim = np.dot(pred_vel, gt_vel) / (pred_vel_norm * gt_vel_norm)
                # Cosine similarity in [-1, 1], map to [0, 1]
                direction_consistency = (cosine_sim + 1) / 2
                consecutive_matches += direction_consistency
            else:
                # Both stationary → perfect association
                consecutive_matches += 1.0

    if total_pairs == 0:
        return 1.0

    ass_a = consecutive_matches / total_pairs
    return ass_a
