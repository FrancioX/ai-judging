"""Evaluate segmentation (detection) quality against ground-truth center annotations.

Compares raw YOLO+ByteTrack detection bbox centers from ``segmentation.json``
against hand-annotated skier centers — **before** any tracking post-processing.
This isolates detection-level problems from tracking smoothing/gap-filling.

Metrics
-------
- **Raw center error**: bbox center vs GT center (mean/median/p90/p95/max).
- **Detection rate**: % of GT frames where any person was detected.
- **Track fragmentation**: how many ByteTrack IDs are assigned to the target.
- **Longest track ratio**: longest continuous correct-ID run / total GT frames.
- **Correct-person rate**: when multiple people detected, how often is the
  closest-to-GT person the one ByteTrack selected as best track.

Usage
-----
::

    # Single video
    uv run python -m src.segmentation.evaluate \\
        "output/segmentation/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_centers.csv"

    # Batch: evaluate all annotated videos
    uv run python -m src.segmentation.evaluate --batch
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
_SEGMENTATION_DIR = _PROJECT_ROOT / "output" / "segmentation"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_center_gt(gt_path: Path) -> dict[int, tuple[float, float]]:
    """Load sparse center-point ground truth from CSV."""
    annotations: dict[int, tuple[float, float]] = {}
    with open(gt_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            frame_id = int(row[0])
            annotations[frame_id] = (float(row[2]), float(row[3]))
    return annotations


def _load_segmentation(seg_dir: Path) -> list[dict]:
    """Load segmentation manifest frames."""
    manifest_path = seg_dir / "segmentation.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Segmentation manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f).get("frames", [])


def _frame_id_from_file(frame_file: str) -> int | None:
    """Extract integer frame id from filename like ``frame_000150.jpg``."""
    try:
        return int(frame_file.replace("frame_", "").replace(".jpg", "").replace(".png", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Interpolation (reused from tracking evaluate)
# ---------------------------------------------------------------------------

def _interpolate_gt(
    gt: dict[int, tuple[float, float]],
    all_frame_ids: list[int],
) -> dict[int, tuple[float, float]]:
    """Linearly interpolate sparse GT centers to cover all frames."""
    if len(gt) < 2:
        return dict(gt)

    keyframes = sorted(gt.keys())
    first_kf, last_kf = keyframes[0], keyframes[-1]

    dense: dict[int, tuple[float, float]] = {}
    for fid in all_frame_ids:
        if fid < first_kf or fid > last_kf:
            continue
        if fid in gt:
            dense[fid] = gt[fid]
            continue
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

def evaluate_segmentation(
    seg_dir: Path,
    gt_path: Path,
    *,
    distance_threshold: float = 50.0,
) -> dict:
    """Evaluate raw segmentation detections against ground truth.

    For each GT frame, finds the closest detected person bbox center and
    measures error. Also computes track fragmentation metrics.
    """
    gt_sparse = _load_center_gt(gt_path)
    seg_frames = _load_segmentation(seg_dir)

    if not gt_sparse:
        raise RuntimeError(f"No annotations found in {gt_path}")

    # Build frame lookup: frame_id -> frame dict
    frame_lookup: dict[int, dict] = {}
    all_frame_ids: list[int] = []
    for fr in seg_frames:
        fid = _frame_id_from_file(fr.get("frame_file", ""))
        if fid is not None:
            frame_lookup[fid] = fr
            all_frame_ids.append(fid)
    all_frame_ids.sort()

    gt = _interpolate_gt(gt_sparse, all_frame_ids)

    # Evaluate each GT frame
    errors: list[float] = []
    best_track_ids: list[int | None] = []
    selected_track_ids: list[int | None] = []
    no_detection_count = 0
    correct_person_count = 0
    multi_person_count = 0

    for fid in sorted(gt.keys()):
        gcx, gcy = gt[fid]
        fr = frame_lookup.get(fid)

        if fr is None or not fr.get("detected", False):
            no_detection_count += 1
            best_track_ids.append(None)
            selected_track_ids.append(None)
            continue

        persons = fr.get("persons", [])
        if not persons:
            no_detection_count += 1
            best_track_ids.append(None)
            selected_track_ids.append(None)
            continue

        # Find closest person to GT
        min_dist = float("inf")
        closest_tid = None
        for p in persons:
            bx1, by1, bx2, by2 = p["bbox"]
            pcx = (bx1 + bx2) / 2.0
            pcy = (by1 + by2) / 2.0
            d = np.sqrt((gcx - pcx) ** 2 + (gcy - pcy) ** 2)
            if d < min_dist:
                min_dist = d
                closest_tid = p.get("track_id")

        errors.append(float(min_dist))
        best_track_ids.append(closest_tid)

        # What did ByteTrack select as the frame's main detection?
        sel_tid = fr.get("track_id")
        selected_track_ids.append(sel_tid)

        # Multi-person correct selection
        if len(persons) > 1:
            multi_person_count += 1
            if sel_tid == closest_tid:
                correct_person_count += 1

    errors_arr = np.array(errors) if errors else np.array([0.0])
    n_evaluated = len(errors) + no_detection_count
    within_threshold = sum(1 for e in errors if e <= distance_threshold)

    # --- Track fragmentation ---
    # Which ByteTrack IDs does the target athlete get across GT frames?
    valid_tids = [t for t in best_track_ids if t is not None]
    unique_tids = set(valid_tids)

    # Longest continuous run of same track ID
    longest_run = 0
    current_run = 0
    prev_tid = None
    for tid in best_track_ids:
        if tid is not None and tid == prev_tid:
            current_run += 1
        else:
            current_run = 1
        if current_run > longest_run:
            longest_run = current_run
        prev_tid = tid

    results = {
        "video": seg_dir.name,
        "gt_keyframes": len(gt_sparse),
        "gt_frames_evaluated": n_evaluated,
        "gt_frames_no_detection": no_detection_count,
        "detection_rate": round((n_evaluated - no_detection_count) / max(n_evaluated, 1) * 100, 1),
        "mean_error_px": round(float(np.mean(errors_arr)), 1),
        "median_error_px": round(float(np.median(errors_arr)), 1),
        "std_error_px": round(float(np.std(errors_arr)), 1),
        "max_error_px": round(float(np.max(errors_arr)), 1),
        "p90_error_px": round(float(np.percentile(errors_arr, 90)), 1),
        "p95_error_px": round(float(np.percentile(errors_arr, 95)), 1),
        "pct_within_threshold": round(within_threshold / max(len(errors), 1) * 100, 1),
        "distance_threshold_px": distance_threshold,
        "unique_track_ids": len(unique_tids),
        "track_id_list": sorted(unique_tids),
        "longest_run_frames": longest_run,
        "longest_run_ratio": round(longest_run / max(len(valid_tids), 1), 3),
        "multi_person_frames": multi_person_count,
        "correct_person_rate": round(correct_person_count / max(multi_person_count, 1) * 100, 1),
    }

    _print_results(results)
    return results


def _print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  SEGMENTATION EVALUATION — {results['video']}")
    print(f"{'=' * 60}")

    keys = [
        "gt_keyframes", "gt_frames_evaluated", "gt_frames_no_detection",
        "detection_rate",
        "mean_error_px", "median_error_px", "std_error_px",
        "max_error_px", "p90_error_px", "p95_error_px",
        "pct_within_threshold", "distance_threshold_px",
    ]
    for k in keys:
        label = k.replace("_", " ").title()
        print(f"  {label:>30s}:  {results[k]}")

    print(f"\n  {'--- Track Fragmentation ---':>30s}")
    print(f"  {'Unique ByteTrack IDs':>30s}:  {results['unique_track_ids']}")
    print(f"  {'Track ID List':>30s}:  {results['track_id_list']}")
    print(f"  {'Longest Run (frames)':>30s}:  {results['longest_run_frames']}")
    print(f"  {'Longest Run Ratio':>30s}:  {results['longest_run_ratio']}")

    print(f"\n  {'--- Multi-Person Selection ---':>30s}")
    print(f"  {'Multi-Person Frames':>30s}:  {results['multi_person_frames']}")
    print(f"  {'Correct Person Rate (%)':>30s}:  {results['correct_person_rate']}")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    annotations_dir: Path | None = None,
    segmentation_dir: Path | None = None,
    **kwargs: float,
) -> list[dict]:
    """Evaluate all videos that have both segmentation output and annotations."""
    ann_root = annotations_dir or _ANNOTATIONS_DIR
    seg_root = segmentation_dir or _SEGMENTATION_DIR

    if not ann_root.exists():
        raise FileNotFoundError(f"Annotations directory not found: {ann_root}")

    results = []
    for gt_csv in sorted(ann_root.rglob("gt_centers.csv")):
        video_stem = gt_csv.parent.name
        seg_dir = seg_root / video_stem
        if not seg_dir.exists():
            print(f"  SKIP {video_stem} — no segmentation output")
            continue
        print(f"\nEvaluating: {video_stem}")
        result = evaluate_segmentation(seg_dir, gt_csv, **kwargs)
        results.append(result)

    if results:
        print(f"\n{'=' * 60}")
        print("  SUMMARY ACROSS ALL VIDEOS")
        print(f"{'=' * 60}")
        mean_errors = [r["mean_error_px"] for r in results]
        det_rates = [r["detection_rate"] for r in results]
        frag = [r["unique_track_ids"] for r in results]
        run_ratios = [r["longest_run_ratio"] for r in results]
        correct_rates = [r["correct_person_rate"] for r in results]

        print(f"  {'Videos evaluated':>30s}:  {len(results)}")
        print(f"  {'Mean error (px)':>30s}:  {np.mean(mean_errors):.1f}")
        print(f"  {'Worst mean error (px)':>30s}:  {np.max(mean_errors):.1f}")
        print(f"  {'Best mean error (px)':>30s}:  {np.min(mean_errors):.1f}")
        print(f"  {'Mean detection rate (%)':>30s}:  {np.mean(det_rates):.1f}")

        print(f"\n  {'--- Fragmentation Summary ---':>30s}")
        print(f"  {'Mean unique track IDs':>30s}:  {np.mean(frag):.1f}")
        print(f"  {'Max unique track IDs':>30s}:  {np.max(frag)}")
        print(f"  {'Mean longest run ratio':>30s}:  {np.mean(run_ratios):.3f}")

        print(f"\n  {'--- Selection Summary ---':>30s}")
        print(f"  {'Mean correct person rate (%)':>30s}:  {np.mean(correct_rates):.1f}")

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
        cli_kwargs: dict = {}
        for arg in sys.argv[1:]:
            if arg.startswith("--threshold="):
                cli_kwargs["distance_threshold"] = float(arg.split("=")[1])
        evaluate_all(**cli_kwargs)
        sys.exit(0)

    if len(sys.argv) < 3:
        print("Usage: uv run python -m src.segmentation.evaluate <seg_dir> <gt_csv> [options]")
        print("       uv run python -m src.segmentation.evaluate --batch")
        sys.exit(1)

    seg_dir_arg = Path(sys.argv[1])
    gt_path_arg = Path(sys.argv[2])

    eval_kwargs: dict = {}
    for arg in sys.argv[3:]:
        if arg.startswith("--threshold="):
            eval_kwargs["distance_threshold"] = float(arg.split("=")[1])

    evaluate_segmentation(seg_dir_arg, gt_path_arg, **eval_kwargs)
