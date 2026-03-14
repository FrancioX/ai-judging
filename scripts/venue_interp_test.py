"""Compare interpolation methods for venue mapping (LOO cross-validation).

Tests: linear, PCHIP, cubic spline, tracking-arc-length, and a hybrid that
uses tracking velocity to detect non-linear segments and adds extra weight.

Usage
-----
::

    uv run python scripts/venue_interp_test.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d


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


def _predict_at_frame_scipy(
    fid_query: int,
    gt_calib: dict[int, tuple[float, float]],
    centers: dict,
    arc: dict,
    method: str,
    venue_w: int,
    venue_h: int,
) -> tuple[float, float] | None:
    """Interpolate using scipy method for frames within GT range."""
    kfs = sorted(gt_calib.keys())
    if not kfs:
        return None
    first, last = kfs[0], kfs[-1]

    # Within range: use scipy interpolator
    if first <= fid_query <= last:
        kf_arr = np.array(kfs, dtype=float)
        vx_arr = np.array([gt_calib[k][0] for k in kfs], dtype=float)
        vy_arr = np.array([gt_calib[k][1] for k in kfs], dtype=float)

        if method == "pchip":
            fx = PchipInterpolator(kf_arr, vx_arr)
            fy = PchipInterpolator(kf_arr, vy_arr)
        elif method == "cubic":
            bc = "not-a-knot" if len(kfs) >= 4 else "natural"
            fx = CubicSpline(kf_arr, vx_arr, bc_type=bc)
            fy = CubicSpline(kf_arr, vy_arr, bc_type=bc)
        elif method == "linear":
            fx = interp1d(kf_arr, vx_arr, kind="linear", bounds_error=False, fill_value="extrapolate")
            fy = interp1d(kf_arr, vy_arr, kind="linear", bounds_error=False, fill_value="extrapolate")
        elif method == "tracking":
            # Arc-length interpolation
            # Map kfs to their arc-length at the nearest tracked frame
            arc_at_kfs = []
            for k in kfs:
                # find nearest tracked frame
                if k in arc:
                    arc_at_kfs.append(arc[k])
                else:
                    near = min(arc.keys(), key=lambda f: abs(f - k))
                    arc_at_kfs.append(arc[near])
            arc_at_kfs = np.array(arc_at_kfs, dtype=float)
            # get arc at query
            if fid_query in arc:
                arc_q = arc[fid_query]
            else:
                near = min(arc.keys(), key=lambda f: abs(f - fid_query))
                arc_q = arc[near]

            # linear interpolation in arc-length space
            lo_i = np.searchsorted(arc_at_kfs, arc_q, side='right') - 1
            lo_i = int(np.clip(lo_i, 0, len(kfs) - 2))
            hi_i = lo_i + 1
            denom = arc_at_kfs[hi_i] - arc_at_kfs[lo_i]
            t = (arc_q - arc_at_kfs[lo_i]) / denom if denom > 0 else 0.5
            t = float(np.clip(t, 0, 1))
            vx = gt_calib[kfs[lo_i]][0] + t * (gt_calib[kfs[hi_i]][0] - gt_calib[kfs[lo_i]][0])
            vy = gt_calib[kfs[lo_i]][1] + t * (gt_calib[kfs[hi_i]][1] - gt_calib[kfs[lo_i]][1])
            return (float(np.clip(vx, 0, venue_w-1)), float(np.clip(vy, 0, venue_h-1)))
        else:
            raise ValueError(f"Unknown method: {method}")

        vx = float(np.clip(fx(fid_query), 0, venue_w-1))
        vy = float(np.clip(fy(fid_query), 0, venue_h-1))
        return (vx, vy)

    # Outside range: fallback linear extrapolation from nearest segment
    common = sorted(set(centers.keys()) & set(gt_calib.keys()))
    if len(common) >= 4 and fid_query in centers:
        src = np.array([[centers[f][0], centers[f][1]] for f in common], dtype=np.float32)
        dst = np.array([[gt_calib[f][0], gt_calib[f][1]] for f in common], dtype=np.float32)
        H, _ = cv2.findHomography(src.reshape(-1,1,2), dst.reshape(-1,1,2), cv2.RANSAC, 20.0)
        if H is not None:
            cx, cy = centers[fid_query]
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(pt, H.astype(np.float32))
            return (float(np.clip(proj[0,0,0], 0, venue_w-1)), float(np.clip(proj[0,0,1], 0, venue_h-1)))

    # Linear extrapolation from endpoints
    if fid_query < first:
        if len(kfs) >= 2:
            k0, k1 = kfs[0], kfs[1]
            t = float(fid_query - k0) / float(k1 - k0) if k1 != k0 else 0
            vx = gt_calib[k0][0] + t*(gt_calib[k1][0]-gt_calib[k0][0])
            vy = gt_calib[k0][1] + t*(gt_calib[k1][1]-gt_calib[k0][1])
            return (float(np.clip(vx,0,venue_w-1)), float(np.clip(vy,0,venue_h-1)))
        return gt_calib[first]

    if len(kfs) >= 2:
        k0, k1 = kfs[-2], kfs[-1]
        t = float(fid_query - k1) / float(k1 - k0) if k1 != k0 else 0
        vx = gt_calib[k1][0] + t*(gt_calib[k1][0]-gt_calib[k0][0])
        vy = gt_calib[k1][1] + t*(gt_calib[k1][1]-gt_calib[k0][1])
        return (float(np.clip(vx,0,venue_w-1)), float(np.clip(vy,0,venue_h-1)))
    return gt_calib[last]


def run_interp_test(video_path: Path, venue_image_path: Path, tracking_root: Path, gt_root: Path) -> None:
    video_stem = video_path.stem
    tracking_manifest = tracking_root / video_stem / "tracking.json"
    gt_path = gt_root / video_stem / "gt_venue.csv"

    venue_bgr = cv2.imread(str(venue_image_path))
    venue_h, venue_w = venue_bgr.shape[:2]

    _, centers = _load_tracking(tracking_manifest)
    gt_all = _load_gt(gt_path)
    arc = _build_arc_len(centers)
    kfs = sorted(gt_all.keys())

    methods = ["linear", "pchip", "cubic", "tracking"]

    print(f"\nInterpolation method comparison — LOO CV — {video_stem}")
    print(f"GT annotations: {len(kfs)}  |  Venue: {venue_w}×{venue_h}\n")

    header = f"{'Frame':>8}  {'gap':>6}  " + "  ".join(f"{m:>12}" for m in methods)
    print(header)
    print("─" * (8 + 6 + 2 + len(methods) * 14 + 4))

    all_errors = {m: [] for m in methods}

    for i, fid in enumerate(kfs):
        gt_calib = {k: v for k, v in gt_all.items() if k != fid}
        gt_true = gt_all[fid]
        gap_before = fid - kfs[i-1] if i > 0 else 0
        gap_after = kfs[i+1] - fid if i < len(kfs)-1 else 0
        gap = max(gap_before, gap_after)

        errors_row = {}
        for m in methods:
            m_key = m
            pred = _predict_at_frame_scipy(fid, gt_calib, centers, arc, m_key, venue_w, venue_h)
            if pred is None:
                err = float("nan")
            else:
                err = float(np.sqrt((pred[0]-gt_true[0])**2 + (pred[1]-gt_true[1])**2))
            errors_row[m] = err
            all_errors[m].append(err)

        best_m = min(errors_row, key=lambda m: errors_row[m] if not np.isnan(errors_row[m]) else 1e9)
        row = f"{fid:>8}  {gap:>6}  "
        for m in methods:
            e = errors_row[m]
            marker = "*" if m == best_m else " "
            row += f"  {e:>10.1f}{marker}"
        print(row)

    print("─" * (8 + 6 + 2 + len(methods) * 14 + 4))
    for stat_name, fn in [("MEAN", np.mean), ("P90", lambda x: np.percentile(x, 90))]:
        row = f"{stat_name:>8}  {'':>6}  "
        for m in methods:
            arr = np.array([e for e in all_errors[m] if not np.isnan(e)])
            row += f"  {fn(arr):>11.1f} "
        print(row)

    w50_row = f"{'w/in50':>8}  {'':>6}  "
    for m in methods:
        arr = np.array([e for e in all_errors[m] if not np.isnan(e)])
        pct = float((arr <= 50).sum() / len(arr) * 100)
        w50_row += f"  {pct:>10.1f}% "
    print(w50_row)
    print()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Venue interpolation method comparison (LOO CV).")
    p.add_argument("video", type=Path)
    p.add_argument("--venue-image", type=Path, default=Path("venue_image.png"))
    p.add_argument("--tracking-root", type=Path, default=Path("output/tracking"))
    p.add_argument("--gt-root", type=Path, default=Path("annotations/venue"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_interp_test(args.video, args.venue_image, args.tracking_root, args.gt_root)


if __name__ == "__main__":
    main()
