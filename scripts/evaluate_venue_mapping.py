"""Evaluate venue-mapping accuracy against hand-annotated ground-truth positions.

Reads ``output/venue_mapping/<stem>/projection.json`` (which stores per-frame
projected venue coordinates) and compares against the sparse GT annotations in
``annotations/venue/<stem>/gt_venue.csv``.  Sparse GT frames are linearly
interpolated, matching the evaluator used for 2D tracking.

GT CSV format (no header required, comment lines with ``#`` are skipped)::

    # frame_id,venue_x,venue_y
    150,1136.0,124.0
    250,1142.0,170.0

Usage
-----
::

    # Single video
    uv run python scripts/evaluate_venue_mapping.py \\
        "output/venue_mapping/Ski Men_2_89_Andreas Bakke" \\
        "annotations/venue/Ski Men_2_89_Andreas Bakke/gt_venue.csv"

    # Batch: all videos with venue annotations
    uv run python scripts/evaluate_venue_mapping.py --batch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ANNOTATIONS_DIR = _PROJECT_ROOT / "annotations" / "venue"
_MAPPING_DIR = _PROJECT_ROOT / "output" / "venue_mapping"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_gt(gt_path: Path) -> dict[int, tuple[float, float]]:
    """Load venue ground-truth from CSV.

    Parameters
    ----------
    gt_path : Path
        CSV with columns ``frame_id,venue_x,venue_y``.

    Returns
    -------
    dict[int, tuple[float, float]]
        ``{frame_id: (venue_x, venue_y)}``.
    """
    gt: dict[int, tuple[float, float]] = {}
    with open(gt_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            frame_id = int(row[0])
            gt[frame_id] = (float(row[1]), float(row[2]))
    return gt


def _load_projected(mapping_dir: Path) -> dict[int, tuple[float, float]]:
    """Load per-frame projected venue positions from ``projection.json``.

    Parameters
    ----------
    mapping_dir : Path
        Path to ``output/venue_mapping/<video_stem>/``.

    Returns
    -------
    dict[int, tuple[float, float]]
        ``{frame_id: (venue_x, venue_y)}``.

    Raises
    ------
    KeyError
        If ``projected_positions`` key is missing from the manifest (need to
        re-run venue mapping with the updated script).
    """
    manifest_path = mapping_dir / "projection.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"projection.json not found: {manifest_path}")

    with open(manifest_path) as f:
        data = json.load(f)

    if "projected_positions" not in data:
        raise KeyError(
            f"'projected_positions' missing from {manifest_path}. "
            "Re-run venue mapping to regenerate: "
            "uv run python scripts/venue_mapping.py <video>"
        )

    return {
        int(entry["frame_id"]): (float(entry["venue_x"]), float(entry["venue_y"]))
        for entry in data["projected_positions"]
    }


# ---------------------------------------------------------------------------
# Interpolation (mirrors src.tracking.evaluate logic)
# ---------------------------------------------------------------------------

def _interpolate_gt(
    gt: dict[int, tuple[float, float]],
    all_frame_ids: list[int],
) -> dict[int, tuple[float, float]]:
    """Linearly interpolate sparse GT to all predicted frame IDs."""
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
        x0, y0 = gt[lo]
        x1, y1 = gt[hi]
        dense[fid] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))

    return dense


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_venue_mapping(
    mapping_dir: Path,
    gt_path: Path,
    *,
    distance_threshold: float = 50.0,
    interpolate: bool = True,
) -> dict:
    """Compute venue-projection error against ground truth.

    Parameters
    ----------
    mapping_dir : Path
        Path to ``output/venue_mapping/<video_stem>/``.
    gt_path : Path
        Path to ground-truth CSV with venue center annotations.
    distance_threshold : float
        Maximum pixel error (in venue image pixels) counted as "within threshold".
    interpolate : bool
        Linearly interpolate between sparse GT keyframes.

    Returns
    -------
    dict
        Metrics: ``mean_error_px``, ``median_error_px``, ``p90_error_px``,
        ``p95_error_px``, ``max_error_px``, ``pct_within_threshold``,
        ``detection_rate``, ``gt_keyframes``, ``gt_frames_evaluated``.
    """
    gt_sparse = _load_gt(gt_path)
    preds = _load_projected(mapping_dir)

    if not gt_sparse:
        raise RuntimeError(f"No annotations found in {gt_path}")

    all_pred_ids = sorted(preds.keys())

    gt = _interpolate_gt(gt_sparse, all_pred_ids) if interpolate else gt_sparse

    common_frames = sorted(set(gt.keys()) & set(preds.keys()))

    if not common_frames:
        raise RuntimeError(
            f"No overlapping frames between GT ({min(gt)}-{max(gt)}) "
            f"and predictions ({min(preds)}-{max(preds)})"
        )

    errors: list[float] = []
    within_threshold = 0

    for fid in common_frames:
        gx, gy = gt[fid]
        px, py = preds[fid]
        dist = float(np.sqrt((gx - px) ** 2 + (gy - py) ** 2))
        errors.append(dist)
        if dist <= distance_threshold:
            within_threshold += 1

    arr = np.array(errors, dtype=np.float64)
    gt_only = set(gt.keys()) - set(preds.keys())

    return {
        "video": mapping_dir.name,
        "gt_keyframes": len(gt_sparse),
        "gt_frames_evaluated": len(common_frames),
        "gt_frames_no_prediction": len(gt_only),
        "mean_error_px": float(arr.mean()),
        "median_error_px": float(np.median(arr)),
        "p90_error_px": float(np.percentile(arr, 90)),
        "p95_error_px": float(np.percentile(arr, 95)),
        "max_error_px": float(arr.max()),
        "pct_within_threshold": float(within_threshold / len(common_frames) * 100),
        "detection_rate": float(len(preds) / max(1, len(preds) + len(gt_only))),
        "distance_threshold_px": distance_threshold,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_results(results: dict) -> None:
    video = results["video"]
    print(f"\n{'─' * 60}")
    print(f"  Video: {video}")
    print(f"{'─' * 60}")
    print(f"  GT keyframes annotated : {results['gt_keyframes']}")
    print(f"  GT frames evaluated    : {results['gt_frames_evaluated']}")
    print(f"  GT frames w/o pred     : {results['gt_frames_no_prediction']}")
    print(f"  Mean error             : {results['mean_error_px']:.1f} px")
    print(f"  Median error           : {results['median_error_px']:.1f} px")
    print(f"  P90 error              : {results['p90_error_px']:.1f} px")
    print(f"  P95 error              : {results['p95_error_px']:.1f} px")
    print(f"  Max error              : {results['max_error_px']:.1f} px")
    print(
        f"  Within {results['distance_threshold_px']:.0f}px threshold "
        f"               : {results['pct_within_threshold']:.1f}%"
    )
    print(f"  Detection rate         : {results['detection_rate'] * 100:.1f}%")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate venue-mapping projection against ground truth."
    )
    parser.add_argument(
        "mapping_dir",
        type=Path,
        nargs="?",
        help="Path to output/venue_mapping/<stem>/ (omit when using --batch).",
    )
    parser.add_argument(
        "gt_path",
        type=Path,
        nargs="?",
        help="Path to annotations/venue/<stem>/gt_venue.csv (omit when using --batch).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Evaluate all videos that have both a venue annotation and a projection.json.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Distance threshold (px) used to compute pct_within_threshold.",
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Evaluate only on annotated keyframes (no interpolation).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    interpolate = not args.no_interpolate

    if args.batch:
        # Find all video stems with GT annotations
        if not _ANNOTATIONS_DIR.exists():
            print(f"[evaluate_venue] No annotations directory: {_ANNOTATIONS_DIR}")
            sys.exit(1)

        stems = [p.name for p in _ANNOTATIONS_DIR.iterdir() if p.is_dir()]
        if not stems:
            print("[evaluate_venue] No annotated videos found.")
            sys.exit(1)

        all_results = []
        for stem in sorted(stems):
            gt_path = _ANNOTATIONS_DIR / stem / "gt_venue.csv"
            mapping_dir = _MAPPING_DIR / stem

            if not gt_path.exists():
                print(f"[evaluate_venue] Skipping {stem!r}: GT file not found.")
                continue
            if not mapping_dir.exists():
                print(f"[evaluate_venue] Skipping {stem!r}: no venue_mapping output.")
                continue

            try:
                results = evaluate_venue_mapping(
                    mapping_dir,
                    gt_path,
                    distance_threshold=args.threshold,
                    interpolate=interpolate,
                )
                _print_results(results)
                all_results.append(results)
            except (FileNotFoundError, KeyError, RuntimeError) as exc:
                print(f"[evaluate_venue] ERROR for {stem!r}: {exc}")

        if len(all_results) > 1:
            mean_errors = [r["mean_error_px"] for r in all_results]
            print(f"\n{'═' * 60}")
            print(f"  AGGREGATE ({len(all_results)} videos)")
            print(f"{'═' * 60}")
            print(f"  Mean of mean errors : {np.mean(mean_errors):.1f} px")
            print(f"  Median mean error   : {np.median(mean_errors):.1f} px")
            print(f"  Best video          : {min(all_results, key=lambda r: r['mean_error_px'])['video']} ({min(mean_errors):.1f} px)")
            print(f"  Worst video         : {max(all_results, key=lambda r: r['mean_error_px'])['video']} ({max(mean_errors):.1f} px)")
            print(f"{'═' * 60}")

    else:
        if not args.mapping_dir or not args.gt_path:
            parser.error("Provide mapping_dir and gt_path, or use --batch.")

        results = evaluate_venue_mapping(
            args.mapping_dir,
            args.gt_path,
            distance_threshold=args.threshold,
            interpolate=interpolate,
        )
        _print_results(results)


if __name__ == "__main__":
    main()
