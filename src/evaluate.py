"""Unified evaluation: detection (segmentation) + tracking side by side.

Runs both stage evaluations for all annotated videos and prints a combined
summary table so you can see where errors originate.

Usage
-----
::

    uv run python -m src.evaluate
    uv run python -m src.evaluate --threshold=80
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.segmentation.evaluate import evaluate_segmentation
from src.tracking.evaluate import evaluate_tracking

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ANNOTATIONS_DIR = _PROJECT_ROOT / "annotations" / "tracking"
_SEGMENTATION_DIR = _PROJECT_ROOT / "output" / "segmentation"
_TRACKING_DIR = _PROJECT_ROOT / "output" / "tracking"


def _short_name(video_stem: str) -> str:
    """Shorten video stem for table display."""
    # Extract the __N_Sport_Name portion
    parts = video_stem.split("__", 1)
    if len(parts) < 2:
        return video_stem[:40]
    return parts[1][:45]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def evaluate_all(*, distance_threshold: float = 50.0) -> None:
    """Run detection + tracking evaluation and print combined table."""
    if not _ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(f"Annotations directory not found: {_ANNOTATIONS_DIR}")

    gt_csvs = sorted(_ANNOTATIONS_DIR.rglob("gt_centers.csv"))
    if not gt_csvs:
        print("No ground-truth annotations found.")
        return

    rows: list[dict] = []

    for gt_csv in gt_csvs:
        video_stem = gt_csv.parent.name
        seg_dir = _SEGMENTATION_DIR / video_stem
        trk_dir = _TRACKING_DIR / video_stem

        has_seg = seg_dir.exists()
        has_trk = trk_dir.exists()

        if not has_seg and not has_trk:
            print(f"  SKIP {_short_name(video_stem)} — no outputs")
            continue

        row: dict = {"video": video_stem, "short": _short_name(video_stem)}

        # --- Detection ---
        if has_seg:
            try:
                seg = evaluate_segmentation(
                    seg_dir, gt_csv, distance_threshold=distance_threshold,
                )
                row["det_error"] = seg["mean_error_px"]
                row["det_rate"] = seg["detection_rate"]
                row["det_pct_within"] = seg["pct_within_threshold"]
                row["det_correct_person"] = seg["correct_person_rate"]
                row["det_track_ids"] = seg["unique_track_ids"]
                row["det_longest_run"] = seg["longest_run_ratio"]
            except Exception as e:
                print(f"  Detection eval failed for {_short_name(video_stem)}: {e}")
                row["det_error"] = None

        # --- Tracking ---
        if has_trk:
            try:
                trk = evaluate_tracking(
                    trk_dir, gt_csv, distance_threshold=distance_threshold,
                )
                row["trk_error"] = trk["mean_error_px"]
                row["trk_pct_within"] = trk["pct_within_threshold"]
                row["trk_hota"] = trk.get("hota")
            except Exception as e:
                print(f"  Tracking eval failed for {_short_name(video_stem)}: {e}")
                row["trk_error"] = None

        rows.append(row)

    if not rows:
        print("No videos evaluated.")
        return

    _print_table(rows, distance_threshold)


def _print_table(rows: list[dict], threshold: float) -> None:
    """Print combined summary table."""
    print(f"\n{'=' * 120}")
    print(f"  COMBINED EVALUATION — Detection vs Tracking  (threshold={threshold}px)")
    print(f"{'=' * 120}")

    # Header
    header = (
        f"  {'Video':<47s}"
        f"{'Det%':>6s}"
        f"{'DetErr':>8s}"
        f"{'Det<Th':>7s}"
        f"{'CorPer':>7s}"
        f"{'IDs':>5s}"
        f"{'LRun':>6s}"
        f"{'TrkErr':>8s}"
        f"{'Trk<Th':>7s}"
        f"{'HOTA':>7s}"
        f"{'Imprv':>7s}"
    )
    print(header)
    print(f"  {'-' * 115}")

    for r in sorted(rows, key=lambda x: x.get("trk_hota") or 0, reverse=True):
        det_err = r.get("det_error")
        trk_err = r.get("trk_error")

        # Improvement: how much tracking reduces error vs raw detection
        if det_err is not None and trk_err is not None and det_err > 0:
            improvement = f"{(1 - trk_err / det_err) * 100:+.0f}%"
        else:
            improvement = "—"

        line = (
            f"  {r['short']:<47s}"
            f"{_fmt(r.get('det_rate'), '%'):>6s}"
            f"{_fmt(r.get('det_error'), 'px'):>8s}"
            f"{_fmt(r.get('det_pct_within'), '%'):>7s}"
            f"{_fmt(r.get('det_correct_person'), '%'):>7s}"
            f"{_fmt_int(r.get('det_track_ids')):>5s}"
            f"{_fmt(r.get('det_longest_run'), ''):>6s}"
            f"{_fmt(r.get('trk_error'), 'px'):>8s}"
            f"{_fmt(r.get('trk_pct_within'), '%'):>7s}"
            f"{_fmt(r.get('trk_hota'), ''):>7s}"
            f"{improvement:>7s}"
        )
        print(line)

    # Aggregates
    print(f"  {'-' * 115}")

    det_errors = [r["det_error"] for r in rows if r.get("det_error") is not None]
    trk_errors = [r["trk_error"] for r in rows if r.get("trk_error") is not None]
    det_rates = [r["det_rate"] for r in rows if r.get("det_rate") is not None]
    det_correct = [r["det_correct_person"] for r in rows if r.get("det_correct_person") is not None]
    trk_hotas = [r["trk_hota"] for r in rows if r.get("trk_hota") is not None]

    avg_line = (
        f"  {'AVERAGE':<47s}"
        f"{np.mean(det_rates):>5.1f}%"  if det_rates else f"{'—':>6s}"
    )
    avg_line = f"  {'AVERAGE':<47s}"
    avg_line += f"{_avg(det_rates, '%'):>6s}"
    avg_line += f"{_avg(det_errors, 'px'):>8s}"
    avg_line += f"{'':>7s}"
    avg_line += f"{_avg(det_correct, '%'):>7s}"
    avg_line += f"{'':>5s}"
    avg_line += f"{'':>6s}"
    avg_line += f"{_avg(trk_errors, 'px'):>8s}"
    avg_line += f"{'':>7s}"
    avg_line += f"{_avg(trk_hotas, ''):>7s}"
    avg_line += f"{'':>7s}"
    print(avg_line)

    print(f"{'=' * 120}")
    print(f"  Videos: {len(rows)}  |  Det%=detection rate  |  DetErr/TrkErr=mean center error")
    print(f"  CorPer=correct person in multi-person frames  |  IDs=ByteTrack fragmentation")
    print(f"  LRun=longest run ratio  |  Imprv=tracking error reduction vs detection")
    print(f"{'=' * 120}\n")


def _fmt(val: float | None, suffix: str) -> str:
    if val is None:
        return "—"
    if suffix == "%":
        return f"{val:.1f}%"
    if suffix == "px":
        return f"{val:.1f}"
    return f"{val:.3f}"


def _fmt_int(val: int | None) -> str:
    return str(val) if val is not None else "—"


def _avg(vals: list[float], suffix: str) -> str:
    if not vals:
        return "—"
    m = float(np.mean(vals))
    return _fmt(m, suffix)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    kwargs: dict = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--threshold="):
            kwargs["distance_threshold"] = float(arg.split("=")[1])
        elif arg in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)

    evaluate_all(**kwargs)
