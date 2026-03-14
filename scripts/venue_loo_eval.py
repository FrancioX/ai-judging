"""Leave-one-out cross-validation for venue mapping.

For each GT annotation, removes it from the calibration set and evaluates
prediction at that frame against the true GT annotation.  This is a more
honest test of interpolation quality, since the evaluation compares against
the actual hand-annotated position rather than a linearly-interpolated proxy.

Also tests: what is the accuracy at frames MIDWAY between GT annotations
(the hardest interpolation case)?

Usage
-----
::

    uv run python scripts/venue_loo_eval.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def _load_tracking(p: Path):
    with p.open() as f:
        d = json.load(f)
    frames = d.get("frames", [])
    centers = {}
    for fr in frames:
        bbox = fr.get("bbox")
        if bbox and len(bbox) == 4:
            fid = int(fr["frame_file"].replace("frame_","").replace(".jpg","").replace(".png",""))
            centers[fid] = ((bbox[0]+bbox[2])*0.5, (bbox[1]+bbox[3])*0.5)
    return frames, centers


def _load_gt(p: Path) -> dict[int, tuple[float, float]]:
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


def _predict_at_frame(
    fid_query: int,
    gt_calib: dict[int, tuple[float, float]],
    centers: dict[int, tuple[float, float]],
    arc: dict[int, float],
    mode: str,
    venue_w: int,
    venue_h: int,
) -> tuple[float, float] | None:
    """Predict venue position for fid_query using gt_calib as calibration."""
    kfs = sorted(gt_calib.keys())
    if not kfs:
        return None
    first, last = kfs[0], kfs[-1]

    # Within calibration range: interpolate
    if first <= fid_query <= last:
        lo = max(k for k in kfs if k <= fid_query)
        hi = min(k for k in kfs if k >= fid_query)
        vx0, vy0 = gt_calib[lo]
        vx1, vy1 = gt_calib[hi]
        if mode == "tracking" and arc and lo in arc and hi in arc and fid_query in arc:
            denom = arc[hi] - arc[lo]
            t = (arc[fid_query] - arc[lo]) / denom if denom > 0 else 0.5
        else:
            t = float(fid_query - lo) / float(hi - lo) if hi != lo else 0.0
        t = float(np.clip(t, 0.0, 1.0))
        vx = vx0 + t * (vx1 - vx0)
        vy = vy0 + t * (vy1 - vy0)
        return (float(np.clip(vx, 0, venue_w-1)), float(np.clip(vy, 0, venue_h-1)))

    # Outside: fallback homography
    common = sorted(set(centers.keys()) & set(gt_calib.keys()))
    if len(common) >= 4 and fid_query in centers:
        src = np.array([[centers[f][0], centers[f][1]] for f in common], dtype=np.float32)
        dst = np.array([[gt_calib[f][0], gt_calib[f][1]] for f in common], dtype=np.float32)
        H, _ = cv2.findHomography(src.reshape(-1,1,2), dst.reshape(-1,1,2), cv2.RANSAC, 20.0)
        if H is not None:
            cx, cy = centers[fid_query]
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(pt, H.astype(np.float32))
            return (
                float(np.clip(proj[0,0,0], 0, venue_w-1)),
                float(np.clip(proj[0,0,1], 0, venue_h-1)),
            )

    # Last resort: nearest GT endpoint
    return gt_calib[last] if fid_query > last else gt_calib[first]


def run_loo_eval(
    video_path: Path,
    venue_image_path: Path = Path("venue_image.png"),
    tracking_root: Path = Path("output/tracking"),
    gt_root: Path = Path("annotations/venue"),
) -> None:
    video_stem = video_path.stem
    tracking_manifest = tracking_root / video_stem / "tracking.json"
    gt_path = gt_root / video_stem / "gt_venue.csv"

    venue_bgr = cv2.imread(str(venue_image_path))
    venue_h, venue_w = venue_bgr.shape[:2]

    _, centers = _load_tracking(tracking_manifest)
    gt_all = _load_gt(gt_path)
    arc = _build_arc_len(centers)

    kfs = sorted(gt_all.keys())
    print(f"\nLeave-one-out cross-validation — {video_stem}")
    print(f"GT annotations: {len(kfs)} frames")
    print(f"Venue: {venue_w}×{venue_h}\n")

    print(f"{'Frame':>8}  {'gap_before':>10}  {'gap_after':>10}  "
          f"{'linear_err':>11}  {'track_err':>11}  {'winner':>8}")
    print("─" * 75)

    lin_errors = []
    trk_errors = []

    for i, fid in enumerate(kfs):
        # Calibration set: all GT except current frame
        gt_calib = {k: v for k, v in gt_all.items() if k != fid}

        pred_lin = _predict_at_frame(fid, gt_calib, centers, arc, "linear", venue_w, venue_h)
        pred_trk = _predict_at_frame(fid, gt_calib, centers, arc, "tracking", venue_w, venue_h)

        gt_xy = gt_all[fid]

        err_lin = float("nan") if pred_lin is None else float(
            np.sqrt((pred_lin[0]-gt_xy[0])**2 + (pred_lin[1]-gt_xy[1])**2)
        )
        err_trk = float("nan") if pred_trk is None else float(
            np.sqrt((pred_trk[0]-gt_xy[0])**2 + (pred_trk[1]-gt_xy[1])**2)
        )

        gap_before = fid - kfs[i-1] if i > 0 else 0
        gap_after = kfs[i+1] - fid if i < len(kfs)-1 else 0

        winner = "TIE"
        if not np.isnan(err_lin) and not np.isnan(err_trk):
            if err_trk < err_lin - 1:
                winner = "TRACK"
            elif err_lin < err_trk - 1:
                winner = "linear"

        lin_errors.append(err_lin)
        trk_errors.append(err_trk)

        print(f"{fid:>8}  {gap_before:>10}  {gap_after:>10}  "
              f"{err_lin:>11.1f}  {err_trk:>11.1f}  {winner:>8}")

    lin_arr = np.array([e for e in lin_errors if not np.isnan(e)])
    trk_arr = np.array([e for e in trk_errors if not np.isnan(e)])

    print("─" * 75)
    print(f"{'MEAN':>8}  {'':>10}  {'':>10}  "
          f"{lin_arr.mean():>11.1f}  {trk_arr.mean():>11.1f}")
    print(f"{'P90':>8}  {'':>10}  {'':>10}  "
          f"{np.percentile(lin_arr, 90):>11.1f}  {np.percentile(trk_arr, 90):>11.1f}")
    print(f"{'MAX':>8}  {'':>10}  {'':>10}  "
          f"{lin_arr.max():>11.1f}  {trk_arr.max():>11.1f}")
    within50_lin = float((lin_arr <= 50).sum() / len(lin_arr) * 100)
    within50_trk = float((trk_arr <= 50).sum() / len(trk_arr) * 100)
    print(f"{'w/in 50':>8}  {'':>10}  {'':>10}  "
          f"{within50_lin:>10.1f}%  {within50_trk:>10.1f}%")
    print()
    print(f"linear wins: {sum(1 for l,t in zip(lin_errors,trk_errors) if not np.isnan(l) and not np.isnan(t) and l < t-1)}")
    print(f"tracking wins: {sum(1 for l,t in zip(lin_errors,trk_errors) if not np.isnan(l) and not np.isnan(t) and t < l-1)}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Leave-one-out venue mapping evaluation.")
    p.add_argument("video", type=Path)
    p.add_argument("--venue-image", type=Path, default=Path("venue_image.png"))
    p.add_argument("--tracking-root", type=Path, default=Path("output/tracking"))
    p.add_argument("--gt-root", type=Path, default=Path("annotations/venue"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_loo_eval(
        video_path=args.video,
        venue_image_path=args.venue_image,
        tracking_root=args.tracking_root,
        gt_root=args.gt_root,
    )


if __name__ == "__main__":
    main()
