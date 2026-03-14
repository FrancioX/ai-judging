"""Leave-one-out CV using the updated v3 interpolation logic (PCHIP + linear trend extrapolation).

Usage
-----
::

    uv run python scripts/venue_loo_eval_v2.py "raw_videos/Ski Men_2_89_Andreas Bakke.mp4"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator


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
    centers: dict,
    arc: dict,
    mode: str,
    venue_w: int,
    venue_h: int,
) -> tuple[float, float] | None:
    kfs = sorted(gt_calib.keys())
    if not kfs:
        return None
    first, last = kfs[0], kfs[-1]

    def _clamp(v: float) -> tuple[float, float]:
        return (float(np.clip(v[0], 0, venue_w-1)), float(np.clip(v[1], 0, venue_h-1)))

    # Within range
    if first <= fid_query <= last:
        if mode == "pchip" and len(kfs) >= 2:
            kf_arr = np.array(kfs, dtype=float)
            fx = PchipInterpolator(kf_arr, [gt_calib[k][0] for k in kfs])
            fy = PchipInterpolator(kf_arr, [gt_calib[k][1] for k in kfs])
            return _clamp((float(fx(fid_query)), float(fy(fid_query))))

        lo = max(k for k in kfs if k <= fid_query)
        hi = min(k for k in kfs if k >= fid_query)
        vx0, vy0 = gt_calib[lo]
        vx1, vy1 = gt_calib[hi]
        if mode == "tracking" and lo in arc and hi in arc and fid_query in arc:
            denom = arc[hi] - arc[lo]
            t = (arc[fid_query] - arc[lo]) / denom if denom > 0 else 0.5
        else:
            t = float(fid_query - lo) / float(hi - lo) if hi != lo else 0.0
        t = float(np.clip(t, 0.0, 1.0))
        return _clamp((vx0 + t*(vx1-vx0), vy0 + t*(vy1-vy0)))

    # Outside range: linear trend extrapolation
    if fid_query < first:
        if len(kfs) >= 2:
            k0, k1 = kfs[0], kfs[1]
            t = float(fid_query - k0) / float(k1 - k0) if k1 != k0 else 0.0
            return _clamp((gt_calib[k0][0] + t*(gt_calib[k1][0]-gt_calib[k0][0]),
                           gt_calib[k0][1] + t*(gt_calib[k1][1]-gt_calib[k0][1])))
        return gt_calib[first]

    if len(kfs) >= 2:
        k0, k1 = kfs[-2], kfs[-1]
        t = float(fid_query - k1) / float(k1 - k0) if k1 != k0 else 0.0
        return _clamp((gt_calib[k1][0] + t*(gt_calib[k1][0]-gt_calib[k0][0]),
                       gt_calib[k1][1] + t*(gt_calib[k1][1]-gt_calib[k0][1])))
    return gt_calib[last]


def run_loo_eval(video_path: Path, venue_image_path: Path, tracking_root: Path, gt_root: Path) -> None:
    video_stem = video_path.stem
    tracking_manifest = tracking_root / video_stem / "tracking.json"
    gt_path = gt_root / video_stem / "gt_venue.csv"

    venue_bgr = cv2.imread(str(venue_image_path))
    venue_h, venue_w = venue_bgr.shape[:2]

    _, centers = _load_tracking(tracking_manifest)
    gt_all = _load_gt(gt_path)
    arc = _build_arc_len(centers)
    kfs = sorted(gt_all.keys())

    modes = ["pchip", "linear"]
    print(f"\nLOO CV v2 (linear-trend extrapolation) — {video_stem}")
    print(f"GT annotations: {len(kfs)}  |  Venue: {venue_w}×{venue_h}\n")
    print(f"{'Frame':>8}  {'gap_before':>10}  {'gap_after':>10}  " +
          "  ".join(f"{m:>12}" for m in modes))
    print("─" * 60)

    all_errors = {m: [] for m in modes}
    for i, fid in enumerate(kfs):
        gt_calib = {k: v for k, v in gt_all.items() if k != fid}
        gt_true = gt_all[fid]
        gap_before = fid - kfs[i-1] if i > 0 else 0
        gap_after = kfs[i+1] - fid if i < len(kfs)-1 else 0
        row = f"{fid:>8}  {gap_before:>10}  {gap_after:>10}"
        for m in modes:
            pred = _predict_at_frame(fid, gt_calib, centers, arc, m, venue_w, venue_h)
            err = float(np.sqrt((pred[0]-gt_true[0])**2 + (pred[1]-gt_true[1])**2)) if pred else float("nan")
            all_errors[m].append(err)
            row += f"  {err:>12.1f}"
        print(row)

    print("─" * 60)
    for stat, fn in [("MEAN", np.mean), ("P90", lambda x: np.percentile(x, 90))]:
        row = f"{stat:>8}  {'':>10}  {'':>10}"
        for m in modes:
            arr = np.array(all_errors[m])
            row += f"  {fn(arr):>12.1f}"
        print(row)

    w50_row = f"{'w/in50':>8}  {'':>10}  {'':>10}"
    for m in modes:
        arr = np.array(all_errors[m])
        pct = float((arr <= 50).sum() / len(arr) * 100)
        w50_row += f"  {pct:>11.1f}%"
    print(w50_row)
    print()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LOO CV v2.")
    p.add_argument("video", type=Path)
    p.add_argument("--venue-image", type=Path, default=Path("venue_image.png"))
    p.add_argument("--tracking-root", type=Path, default=Path("output/tracking"))
    p.add_argument("--gt-root", type=Path, default=Path("annotations/venue"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_loo_eval(args.video, args.venue_image, args.tracking_root, args.gt_root)


if __name__ == "__main__":
    main()
