"""Evaluate tracking accuracy against ground-truth center-point annotations.

Compares the center of each predicted bounding box (from ``tracking.json``)
against hand-annotated skier centers, using Euclidean pixel distance.

Sparse ground-truth annotations are linearly interpolated to cover all
frames, so you only need to annotate every N-th frame.

Usage
-----
::

    uv run python -m src.tracking.evaluate \\
        "output/tracking/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_centers.csv"

    # With options
    uv run python -m src.tracking.evaluate \\
        "output/tracking/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_centers.csv" \\
        --threshold=80 --no-interpolate

    # Batch: evaluate all annotated videos
    uv run python -m src.tracking.evaluate --batch
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ANNOTATIONS_DIR = _PROJECT_ROOT / "annotations" / "tracking"
_TRACKING_DIR = _PROJECT_ROOT / "output" / "tracking"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_center_gt(gt_path: Path) -> dict[int, tuple[float, float]]:
    """Load center-point ground truth from CSV.

    Expected format::

        # frame_id,track_id,center_x,center_y
        150,1,808.0,84.0

    Parameters
    ----------
    gt_path : Path
        Path to the annotation CSV.

    Returns
    -------
    dict[int, tuple[float, float]]
        ``{frame_id: (center_x, center_y)}``.
    """
    annotations: dict[int, tuple[float, float]] = {}
    with open(gt_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            frame_id = int(row[0])
            annotations[frame_id] = (float(row[2]), float(row[3]))
    return annotations


def _load_tracking_centers(tracking_dir: Path) -> dict[int, tuple[float, float]]:
    """Extract predicted bbox centers from tracking manifest.

    Reads ``tracking.json`` where each frame has ``bbox: [x1, y1, x2, y2]``.

    Parameters
    ----------
    tracking_dir : Path
        Path to ``output/tracking/<video_stem>/``.

    Returns
    -------
    dict[int, tuple[float, float]]
        ``{frame_id: (center_x, center_y)}``, keyed by the ``frame_file``
        integer (e.g. ``frame_000150.jpg`` → 150).
    """
    manifest_path = tracking_dir / "tracking.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Tracking manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    predictions: dict[int, tuple[float, float]] = {}
    for frame in manifest.get("frames", []):
        # Use the frame filename to extract the frame number, which matches
        # the ground-truth frame_id from the annotation tool.
        frame_file = frame.get("frame_file", "")
        try:
            frame_id = int(frame_file.replace("frame_", "").replace(".jpg", "").replace(".png", ""))
        except ValueError:
            continue

        bbox = frame.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            predictions[frame_id] = (cx, cy)

    return predictions


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _interpolate_gt(
    gt: dict[int, tuple[float, float]],
    all_frame_ids: list[int],
) -> dict[int, tuple[float, float]]:
    """Linearly interpolate sparse ground-truth centers to cover all frames.

    Parameters
    ----------
    gt : dict[int, tuple[float, float]]
        Sparse annotations (only the frames you clicked on).
    all_frame_ids : list[int]
        Every frame_id present in predictions.

    Returns
    -------
    dict[int, tuple[float, float]]
        Dense annotations covering frames between the first and last GT
        keyframe.
    """
    if len(gt) < 2:
        return dict(gt)

    keyframes = sorted(gt.keys())
    first_kf, last_kf = keyframes[0], keyframes[-1]

    dense: dict[int, tuple[float, float]] = {}
    for fid in all_frame_ids:
        if fid < first_kf or fid > last_kf:
            continue  # outside annotated range — skip

        if fid in gt:
            dense[fid] = gt[fid]
            continue

        # Find bracketing keyframes
        lo = max(k for k in keyframes if k <= fid)
        hi = min(k for k in keyframes if k >= fid)
        if lo == hi:
            dense[fid] = gt[lo]
            continue

        t = (fid - lo) / (hi - lo)
        cx0, cy0 = gt[lo]
        cx1, cy1 = gt[hi]
        dense[fid] = (cx0 + t * (cx1 - cx0), cy0 + t * (cy1 - cy0))

    return dense


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_tracking(
    tracking_dir: Path,
    gt_path: Path,
    *,
    distance_threshold: float = 50.0,
    interpolate: bool = True,
) -> dict:
    """Compute center-point tracking error against ground truth.

    Parameters
    ----------
    tracking_dir : Path
        Path to ``output/tracking/<video_stem>/``.
    gt_path : Path
        Path to ground-truth CSV with center annotations.
    distance_threshold : float
        Maximum pixel distance for a frame to be counted as "detected".
        Default 50 px for 1080p footage.
    interpolate : bool
        Linearly interpolate between sparse GT keyframes.

    Returns
    -------
    dict
        Evaluation metrics: ``mean_error_px``, ``median_error_px``,
        ``max_error_px``, ``pct_within_threshold``, ``detection_rate``,
        etc.
    """
    gt_sparse = _load_center_gt(gt_path)
    preds = _load_tracking_centers(tracking_dir)

    if not gt_sparse:
        raise RuntimeError(f"No annotations found in {gt_path}")

    all_pred_ids = sorted(preds.keys())

    if interpolate:
        gt = _interpolate_gt(gt_sparse, all_pred_ids)
    else:
        gt = gt_sparse

    # Only evaluate frames where both GT and prediction exist
    common_frames = sorted(set(gt.keys()) & set(preds.keys()))

    if not common_frames:
        raise RuntimeError(
            f"No overlapping frames between GT ({min(gt)}-{max(gt)}) "
            f"and predictions ({min(preds)}-{max(preds)})"
        )

    errors = []
    within_threshold = 0

    for fid in common_frames:
        gcx, gcy = gt[fid]
        pcx, pcy = preds[fid]
        dist = np.sqrt((gcx - pcx) ** 2 + (gcy - pcy) ** 2)
        errors.append(float(dist))
        if dist <= distance_threshold:
            within_threshold += 1

    errors_arr = np.array(errors)

    # Frames with GT but no prediction
    gt_only = set(gt.keys()) - set(preds.keys())

    results = {
        "video": tracking_dir.name,
        "gt_keyframes": len(gt_sparse),
        "gt_frames_evaluated": len(common_frames),
        "gt_frames_no_prediction": len(gt_only),
        "mean_error_px": round(float(np.mean(errors_arr)), 1),
        "median_error_px": round(float(np.median(errors_arr)), 1),
        "std_error_px": round(float(np.std(errors_arr)), 1),
        "max_error_px": round(float(np.max(errors_arr)), 1),
        "p90_error_px": round(float(np.percentile(errors_arr, 90)), 1),
        "p95_error_px": round(float(np.percentile(errors_arr, 95)), 1),
        "pct_within_threshold": round(within_threshold / len(common_frames) * 100, 1),
        "detection_rate": round(len(common_frames) / (len(common_frames) + len(gt_only)) * 100, 1),
        "distance_threshold_px": distance_threshold,
    }

    _print_results(results)
    return results


def _print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  TRACKING EVALUATION — {results['video']}")
    print(f"{'=' * 60}")
    for k, v in results.items():
        if k == "video":
            continue
        label = k.replace("_", " ").title()
        print(f"  {label:>30s}:  {v}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    annotations_dir: Path | None = None,
    tracking_dir: Path | None = None,
    **kwargs: float | bool,
) -> list[dict]:
    """Evaluate all videos that have both tracking output and annotations.

    Parameters
    ----------
    annotations_dir : Path | None
        Root of annotations (default: ``annotations/tracking/``).
    tracking_dir : Path | None
        Root of tracking outputs (default: ``output/tracking/``).
    **kwargs
        Forwarded to :func:`evaluate_tracking`.

    Returns
    -------
    list[dict]
        List of result dicts, one per video.
    """
    ann_root = annotations_dir or _ANNOTATIONS_DIR
    trk_root = tracking_dir or _TRACKING_DIR

    if not ann_root.exists():
        raise FileNotFoundError(f"Annotations directory not found: {ann_root}")

    results = []
    for gt_csv in sorted(ann_root.rglob("gt_centers.csv")):
        video_stem = gt_csv.parent.name
        trk_dir = trk_root / video_stem
        if not trk_dir.exists():
            print(f"  SKIP {video_stem} — no tracking output")
            continue
        print(f"\nEvaluating: {video_stem}")
        result = evaluate_tracking(trk_dir, gt_csv, **kwargs)
        results.append(result)

    if results:
        print(f"\n{'=' * 60}")
        print("  SUMMARY ACROSS ALL VIDEOS")
        print(f"{'=' * 60}")
        mean_errors = [r["mean_error_px"] for r in results]
        print(f"  {'Videos evaluated':>30s}:  {len(results)}")
        print(f"  {'Overall mean error (px)':>30s}:  {np.mean(mean_errors):.1f}")
        print(f"  {'Worst mean error (px)':>30s}:  {np.max(mean_errors):.1f}")
        print(f"  {'Best mean error (px)':>30s}:  {np.min(mean_errors):.1f}")
        print(f"{'=' * 60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    if "--batch" in sys.argv:
        kwargs: dict = {}
        for arg in sys.argv[1:]:
            if arg.startswith("--threshold="):
                kwargs["distance_threshold"] = float(arg.split("=")[1])
            elif arg == "--no-interpolate":
                kwargs["interpolate"] = False
        evaluate_all(**kwargs)
        sys.exit(0)

    if len(sys.argv) < 3:
        print("Usage: uv run python -m src.tracking.evaluate <tracking_dir> <gt_csv> [options]")
        print("       uv run python -m src.tracking.evaluate --batch")
        sys.exit(1)

    tracking_dir = Path(sys.argv[1])
    gt_path = Path(sys.argv[2])

    eval_kwargs: dict = {}
    for arg in sys.argv[3:]:
        if arg.startswith("--threshold="):
            eval_kwargs["distance_threshold"] = float(arg.split("=")[1])
        elif arg == "--no-interpolate":
            eval_kwargs["interpolate"] = False

    evaluate_tracking(tracking_dir, gt_path, **eval_kwargs)
