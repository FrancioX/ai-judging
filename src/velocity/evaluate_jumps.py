"""Evaluate jump detection accuracy against hand-annotated ground truth.

Compares detected jump intervals (from ``velocity.json``) against GT annotations
(from ``annotations/tracking/<stem>/gt_jumps.csv``).

Matching uses temporal IoU: a prediction is a true positive if its IoU with any
GT jump exceeds *iou_threshold* (default 0.3).

Metrics per video:
    precision, recall, F1
    mean IoU of matched pairs
    mean boundary error (start + end, in frames)

Usage
-----
::

    # Single video
    uv run python -m src.velocity.evaluate_jumps \\
        "output/velocity/Ski Men_2_89_Andreas Bakke" \\
        "annotations/tracking/Ski Men_2_89_Andreas Bakke/gt_jumps.csv"

    # Batch: all annotated videos with gt_jumps.csv
    uv run python -m src.velocity.evaluate_jumps --batch
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ANNOTATIONS_DIR = _PROJECT_ROOT / "annotations" / "tracking"
_VELOCITY_DIR = _PROJECT_ROOT / "output" / "velocity"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temporal_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    union = max(a_end, b_end) - min(a_start, b_start) + 1
    return inter / union if union > 0 else 0.0


def _load_gt(gt_path: Path) -> list[dict]:
    """Load gt_jumps.csv → list of {start, end} dicts (tracking frame indices)."""
    jumps = []
    with open(gt_path, newline="") as f:
        for row in csv.DictReader(f):
            s, e = int(row["start_frame"]), int(row["end_frame"])
            if e > s:  # skip empty/placeholder rows
                jumps.append({"start": s, "end": e})
    return jumps


def _load_pred(velocity_path: Path) -> list[dict]:
    """Load velocity.json → list of {start, end} dicts."""
    with open(velocity_path) as f:
        data = json.load(f)
    return [
        {"start": j["start_frame_id"], "end": j["end_frame_id"]}
        for j in data.get("jumps", [])
    ]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_jumps(
    velocity_dir: str | Path,
    gt_path: str | Path,
    iou_threshold: float = 0.3,
    boundary_tolerance: int = 0,
) -> dict:
    """Compare predicted jumps vs GT for one video.

    Parameters
    ----------
    boundary_tolerance : int
        Expand each GT interval by this many frames on each side before
        computing IoU for matching purposes. Actual boundary errors are
        still reported against the original GT boundaries.

    Returns a dict with keys:
        n_gt, n_pred,
        tp, fp, fn,
        precision, recall, f1,
        mean_iou,           # avg IoU of matched (TP) pairs (using expanded GT)
        mean_start_err,     # avg |pred_start - gt_start| of matched pairs (frames)
        mean_end_err,       # avg |pred_end   - gt_end|   of matched pairs (frames)
        matches             # list of (gt_idx, pred_idx, iou, start_err, end_err)
    """
    velocity_dir = Path(velocity_dir)
    velocity_path = velocity_dir / "velocity.json"
    gt_path = Path(gt_path)

    gt = _load_gt(gt_path)
    pred = _load_pred(velocity_path)

    n_gt = len(gt)
    n_pred = len(pred)

    # Build IoU matrix
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches = []

    # Greedy matching: sort by IoU descending, using tolerance-expanded GT
    candidates = []
    for gi, g in enumerate(gt):
        g_start = g["start"] - boundary_tolerance
        g_end = g["end"] + boundary_tolerance
        for pi, p in enumerate(pred):
            iou = _temporal_iou(g_start, g_end, p["start"], p["end"])
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True)
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        start_err = abs(pred[pi]["start"] - gt[gi]["start"])
        end_err = abs(pred[pi]["end"] - gt[gi]["end"])
        matches.append((gi, pi, iou, start_err, end_err))

    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    mean_iou = sum(m[2] for m in matches) / tp if tp else 0.0
    mean_start_err = sum(m[3] for m in matches) / tp if tp else 0.0
    mean_end_err = sum(m[4] for m in matches) / tp if tp else 0.0

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "mean_iou": round(mean_iou, 3),
        "mean_start_err": round(mean_start_err, 1),
        "mean_end_err": round(mean_end_err, 1),
        "matches": matches,
    }


def _print_report(stem: str, m: dict, gt: list[dict], pred: list[dict], iou_threshold: float) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {stem}")
    print(f"{'─' * 60}")
    print(f"  GT jumps: {m['n_gt']}   Predicted: {m['n_pred']}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  F1={m['f1']:.3f}")
    print(f"  Mean IoU (matched): {m['mean_iou']:.3f}")
    print(f"  Mean boundary err : start={m['mean_start_err']:.1f} fr  end={m['mean_end_err']:.1f} fr")

    if m["matches"]:
        print(f"\n  Matched pairs (IoU ≥ {iou_threshold}):")
        print(f"  {'GT':>4}  {'Pred':>4}  {'IoU':>5}  {'Δstart':>6}  {'Δend':>6}  GT interval      Pred interval")
        for gi, pi, iou, se, ee in m["matches"]:
            g = gt[gi]
            p = pred[pi]
            print(
                f"  {gi+1:>4}  {pi+1:>4}  {iou:>5.3f}  {se:>+6}  {ee:>+6}"
                f"  [{g['start']:>4}–{g['end']:>4}]  [{p['start']:>4}–{p['end']:>4}]"
            )

    matched_gt_idxs = {gi for gi, _, _, _, _ in m["matches"]}
    matched_pred_idxs = {pi for _, pi, _, _, _ in m["matches"]}

    unmatched_gt = [i for i in range(m["n_gt"]) if i not in matched_gt_idxs]
    if unmatched_gt:
        print(f"\n  Missed GT jumps (FN):")
        for i in unmatched_gt:
            g = gt[i]
            print(f"    GT {i+1}: [{g['start']}–{g['end']}]  ({g['end']-g['start']+1} frames)")

    unmatched_pred = [i for i in range(m["n_pred"]) if i not in matched_pred_idxs]
    if unmatched_pred:
        print(f"\n  False positive predictions (FP):")
        for i in unmatched_pred:
            p = pred[i]
            print(f"    Pred {i+1}: [{p['start']}–{p['end']}]  ({p['end']-p['start']+1} frames)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _batch(iou_threshold: float = 0.3, boundary_tolerance: int = 0) -> None:
    results = []
    for gt_path in sorted(_ANNOTATIONS_DIR.glob("*/gt_jumps.csv")):
        stem = gt_path.parent.name
        velocity_path = _VELOCITY_DIR / stem / "velocity.json"
        if not velocity_path.exists():
            print(f"  [skip] No velocity.json for: {stem}")
            continue
        gt = _load_gt(gt_path)
        pred = _load_pred(velocity_path)
        if not gt:
            print(f"  [skip] Empty GT for: {stem}")
            continue
        m = evaluate_jumps(_VELOCITY_DIR / stem, gt_path, iou_threshold=iou_threshold, boundary_tolerance=boundary_tolerance)
        _print_report(stem, m, gt, pred, iou_threshold)
        results.append((stem, m))

    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("  AGGREGATE")
        print(f"{'=' * 60}")
        total_tp = sum(r["tp"] for _, r in results)
        total_fp = sum(r["fp"] for _, r in results)
        total_fn = sum(r["fn"] for _, r in results)
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        mean_iou = sum(r["mean_iou"] for _, r in results) / len(results)
        print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  Mean IoU={mean_iou:.3f}")
        print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]

    iou_threshold = 0.3
    boundary_tolerance = 0
    for a in argv:
        if a.startswith("--iou-threshold="):
            iou_threshold = float(a.split("=")[1])
        elif a.startswith("--boundary-tolerance="):
            boundary_tolerance = int(a.split("=")[1])

    if "--batch" in argv:
        _batch(iou_threshold=iou_threshold, boundary_tolerance=boundary_tolerance)
        return

    if len(argv) < 2 or argv[0].startswith("--"):
        print("Usage: python -m src.velocity.evaluate_jumps <velocity_dir> <gt_jumps.csv> [--iou-threshold=0.3] [--boundary-tolerance=0]")
        sys.exit(1)

    velocity_dir = Path(argv[0])
    gt_path = Path(argv[1])

    stem = velocity_dir.name
    gt = _load_gt(gt_path)
    pred = _load_pred(velocity_path=velocity_dir / "velocity.json")
    m = evaluate_jumps(velocity_dir, gt_path, iou_threshold=iou_threshold, boundary_tolerance=boundary_tolerance)
    _print_report(stem, m, gt, pred, iou_threshold)


if __name__ == "__main__":
    main()
