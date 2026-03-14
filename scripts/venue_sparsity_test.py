"""Venue mapping sparsity test.

Evaluates how accuracy degrades as GT annotations are made sparser.
For each subset size N (1, 2, 3, 5, 7, 10, 15), selects the N most
evenly-spaced GT frames, runs the v3 mapping, and evaluates.

This simulates the calibration effort: how many annotations does a user
need per video to achieve acceptable accuracy?

Usage
-----
::

    uv run python scripts/venue_sparsity_test.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Inline the v3 helpers to avoid import path issues
import cv2


def _load_tracking(p: Path):
    with p.open() as f:
        d = json.load(f)
    frames = d.get("frames", [])
    centers = {}
    for fr in frames:
        bbox = fr.get("bbox")
        if bbox and len(bbox) == 4:
            fid = int(fr["frame_file"].replace("frame_", "").replace(".jpg","").replace(".png",""))
            centers[fid] = ((bbox[0]+bbox[2])*0.5, (bbox[1]+bbox[3])*0.5)
    return frames, centers


def _load_gt(p: Path):
    gt = {}
    with p.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                gt[int(row[0])] = (float(row[1]), float(row[2]))
            except (ValueError, IndexError):
                continue
    return gt


def _interp_gt_dense(gt_sparse: dict, all_frame_ids: list) -> dict:
    """Mirror evaluator: linearly interpolate sparse GT to all pred frame IDs."""
    if len(gt_sparse) < 2:
        return dict(gt_sparse)
    kfs = sorted(gt_sparse.keys())
    first, last = kfs[0], kfs[-1]
    dense = {}
    for fid in all_frame_ids:
        if fid < first or fid > last:
            continue
        if fid in gt_sparse:
            dense[fid] = gt_sparse[fid]
            continue
        lo = max(k for k in kfs if k <= fid)
        hi = min(k for k in kfs if k >= fid)
        t = (fid - lo) / (hi - lo) if hi != lo else 0.0
        x0, y0 = gt_sparse[lo]
        x1, y1 = gt_sparse[hi]
        dense[fid] = (x0 + t*(x1-x0), y0 + t*(y1-y0))
    return dense


def _build_arc_len(centers: dict) -> dict:
    arc = {}
    cum = 0.0
    prev = None
    for fid in sorted(centers):
        if prev is not None:
            dx = centers[fid][0] - centers[prev][0]
            dy = centers[fid][1] - centers[prev][1]
            cum += (dx*dx + dy*dy)**0.5
        arc[fid] = cum
        prev = fid
    return arc


def _project_dense(frame_ids: list, centers: dict, gt_calib: dict,
                   venue_w: int, venue_h: int, mode: str, arc: dict | None = None) -> dict:
    """Produce venue predictions for all frame_ids using calibration GT."""
    kfs = sorted(gt_calib.keys())
    if not kfs:
        return {}
    first, last = kfs[0], kfs[-1]

    # Fit fallback homography
    common = sorted(set(centers.keys()) & set(gt_calib.keys()))
    H = None
    if len(common) >= 4:
        src = np.array([[centers[f][0], centers[f][1]] for f in common], dtype=np.float32)
        dst = np.array([[gt_calib[f][0], gt_calib[f][1]] for f in common], dtype=np.float32)
        H_raw, _ = cv2.findHomography(src.reshape(-1,1,2), dst.reshape(-1,1,2), cv2.RANSAC, 20.0)
        if H_raw is not None:
            H = H_raw.astype(np.float32)

    preds = {}
    for fid in frame_ids:
        if fid in gt_calib:
            preds[fid] = gt_calib[fid]
            continue

        if first <= fid <= last:
            lo = max(k for k in kfs if k <= fid)
            hi = min(k for k in kfs if k >= fid)
            vx0, vy0 = gt_calib[lo]
            vx1, vy1 = gt_calib[hi]
            if mode == "tracking" and arc and lo in arc and hi in arc and fid in arc:
                denom = arc[hi] - arc[lo]
                t = (arc[fid] - arc[lo]) / denom if denom > 0 else 0.5
            else:
                t = float(fid - lo) / float(hi - lo) if hi != lo else 0.0
            t = float(np.clip(t, 0.0, 1.0))
            preds[fid] = (vx0 + t*(vx1-vx0), vy0 + t*(vy1-vy0))
        elif H is not None and fid in centers:
            cx, cy = centers[fid]
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(pt, H)
            preds[fid] = (
                float(np.clip(proj[0,0,0], 0, venue_w-1)),
                float(np.clip(proj[0,0,1], 0, venue_h-1)),
            )
        else:
            ep = gt_calib[last] if fid > last else gt_calib[first]
            preds[fid] = ep

    return preds


def _evaluate(preds: dict, gt_dense: dict) -> dict:
    common = sorted(set(preds.keys()) & set(gt_dense.keys()))
    if not common:
        return {"mean": float("nan"), "p90": float("nan"), "within50": 0.0, "n": 0}
    errs = []
    within50 = 0
    for fid in common:
        dx = preds[fid][0] - gt_dense[fid][0]
        dy = preds[fid][1] - gt_dense[fid][1]
        e = (dx*dx + dy*dy)**0.5
        errs.append(e)
        if e <= 50:
            within50 += 1
    arr = np.array(errs)
    return {
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "within50": float(within50 / len(common) * 100),
        "n": len(common),
    }


def _evenly_spaced_subset(keys: list[int], n: int) -> list[int]:
    """Select n evenly-spaced items from a sorted list."""
    if n >= len(keys):
        return keys
    if n == 1:
        return [keys[len(keys) // 2]]
    indices = [round(i * (len(keys) - 1) / (n - 1)) for i in range(n)]
    return [keys[i] for i in indices]


def run_sparsity_test(
    video_path: Path,
    venue_image_path: Path = Path("venue_image.png"),
    tracking_root: Path = Path("output/tracking"),
    gt_root: Path = Path("annotations/venue"),
) -> None:
    video_stem = video_path.stem
    tracking_manifest = tracking_root / video_stem / "tracking.json"
    gt_path = gt_root / video_stem / "gt_venue.csv"

    for p, label in [(tracking_manifest, "Tracking"), (gt_path, "GT"), (venue_image_path, "Venue")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    venue_bgr = cv2.imread(str(venue_image_path))
    venue_h, venue_w = venue_bgr.shape[:2]

    tracking_frames, centers = _load_tracking(tracking_manifest)
    gt_all = _load_gt(gt_path)
    arc = _build_arc_len(centers)

    frame_ids = [
        int(fr["frame_file"].replace("frame_","").replace(".jpg","").replace(".png",""))
        for fr in tracking_frames
    ]

    # Full dense GT (evaluator-style linear interpolation)
    gt_dense = _interp_gt_dense(gt_all, frame_ids)

    all_kfs = sorted(gt_all.keys())
    subset_sizes = [1, 2, 3, 5, 7, 10, len(all_kfs)]

    print(f"\nSparsity test — {video_stem}")
    print(f"Total GT annotations: {len(all_kfs)} (frames: {all_kfs})")
    print(f"Tracking frames: {len(frame_ids)}  |  Venue: {venue_w}×{venue_h}")
    print()
    print(f"{'N annot':>8}  {'linear mean':>12}  {'linear p90':>11}  {'linear w50':>11}  "
          f"{'track mean':>11}  {'track p90':>10}  {'track w50':>10}")
    print("─" * 90)

    for n in subset_sizes:
        kf_subset = _evenly_spaced_subset(all_kfs, n)
        gt_calib = {fid: gt_all[fid] for fid in kf_subset}

        preds_lin = _project_dense(frame_ids, centers, gt_calib, venue_w, venue_h, "linear")
        preds_trk = _project_dense(frame_ids, centers, gt_calib, venue_w, venue_h, "tracking", arc)

        r_lin = _evaluate(preds_lin, gt_dense)
        r_trk = _evaluate(preds_trk, gt_dense)

        n_label = f"{n} ({', '.join(str(f) for f in kf_subset)})" if n <= 3 else str(n)
        print(
            f"{n:>8}  {r_lin['mean']:>12.1f}  {r_lin['p90']:>11.1f}  {r_lin['within50']:>10.1f}%  "
            f"{r_trk['mean']:>11.1f}  {r_trk['p90']:>10.1f}  {r_trk['within50']:>10.1f}%"
        )

    print()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Venue sparsity test.")
    p.add_argument("video", type=Path)
    p.add_argument("--venue-image", type=Path, default=Path("venue_image.png"))
    p.add_argument("--tracking-root", type=Path, default=Path("output/tracking"))
    p.add_argument("--gt-root", type=Path, default=Path("annotations/venue"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_sparsity_test(
        video_path=args.video,
        venue_image_path=args.venue_image,
        tracking_root=args.tracking_root,
        gt_root=args.gt_root,
    )


if __name__ == "__main__":
    main()
