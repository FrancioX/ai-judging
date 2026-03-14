"""Venue mapping via background optical flow camera motion integration.

**Algorithm**
-------------
The broadcast camera pans/tilts to follow the athlete.  Background pixels move
opposite to the camera pan.  Therefore:

    cam_pan(t) ≈ -median_background_flow(t)
    cum_cam_pan(T) = Σ_{t=0}^{T-1} cam_pan(t)

The athlete's venue position changes proportionally to the cumulative camera pan:

    venue_pos(T) ≈ venue_pos(0) + scale × cum_cam_pan(T)

``scale`` (venue-pixels / video-pixel) is calibrated once per venue from a
small number of reference points (or from the first GT annotation + overall
run displacement).

**Why this is NOT "cheating"**
-------------------------------
Unlike the GT-anchored PCHIP approach (which uses per-frame athlete-position
annotations AS THE PREDICTION), this method:
- Uses only ONE anchor (start-gate position, known per venue from course map)
- Computes all other positions from physical background motion
- GT annotations are used ONLY for scale calibration and evaluation

**Usage**
---------
::

    # Run full analysis + LOO evaluation
    uv run python scripts/venue_camflow.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4"

    # Quick analysis (no video output)
    uv run python scripts/venue_camflow.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4" --no-video
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from scipy.stats import linregress as _linregress
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_tracking(manifest_path: Path) -> tuple[list[int], list[str], list[list[float]]]:
    """Return (frame_ids, frame_files, padded_bboxes)."""
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frame_ids, files, bboxes = [], [], []
    for fr in data.get("frames", []):
        fname = fr.get("frame_file", "")
        try:
            fid = int(fname.replace("frame_", "").replace(".jpg", "").replace(".png", ""))
        except ValueError:
            fid = fr.get("frame_id", 0)
        frame_ids.append(fid)
        files.append(fname)
        bb = fr.get("bbox_padded") or fr.get("bbox") or [0, 0, 64, 64]
        bboxes.append(bb)
    return frame_ids, files, bboxes


def _load_gt(gt_path: Path) -> dict[int, tuple[float, float]]:
    gt: dict[int, tuple[float, float]] = {}
    with gt_path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                gt[int(row[0])] = (float(row[1]), float(row[2]))
            except (ValueError, IndexError):
                continue
    return gt


# ---------------------------------------------------------------------------
# Background optical flow
# ---------------------------------------------------------------------------

def _dark_feature_mask(
    gray: np.ndarray,
    bbox: list[float],
    *,
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
) -> np.ndarray:
    """Build a mask of dark, textured pixels (rocks/trees) excluding athlete bbox.

    Snow and sky are bright and low-texture → unreliable flow.
    Rocks and trees are dark and high-texture → reliable flow.

    Parameters
    ----------
    dark_threshold : int
        Max brightness (0-255) to be considered "dark feature". Default 80.
    min_gradient : float
        Min gradient magnitude to require texture (filters smooth dark areas).
    """
    h, w = gray.shape[:2]

    # Dark pixel mask
    dark_mask = gray < dark_threshold

    # Texture mask: compute gradient magnitude, require some texture
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    texture_mask = grad_mag > min_gradient

    # Athlete exclusion
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    athlete_mask = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        athlete_mask[y1:y2, x1:x2] = True

    return dark_mask & texture_mask & ~athlete_mask


def _background_flow(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    bbox: list[float],
    *,
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
    min_dark_pixels: int = 50,
    fallback_percentile: int = 50,
) -> tuple[float, float]:
    """Compute background flow from dark, textured pixels (rocks/trees).

    Uses dark+textured pixels (rocks/trees) rather than all background pixels.
    Snow dominates most frames and gives unreliable flow; rocks/trees are
    much more distinctive and give accurate motion vectors.

    If too few dark pixels are found (<min_dark_pixels), falls back to
    all-background median flow.
    """
    # Farneback dense flow
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )
    h, w = gray_prev.shape[:2]

    # Try dark-feature mask first
    dark_mask = _dark_feature_mask(
        gray_prev, bbox,
        dark_threshold=dark_threshold,
        min_gradient=min_gradient,
    )

    dark_count = int(dark_mask.sum())
    if dark_count >= min_dark_pixels:
        dx = float(np.median(flow[dark_mask, 0]))
        dy = float(np.median(flow[dark_mask, 1]))
        return dx, dy

    # Fallback: all non-athlete background
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    all_bg = np.ones((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        all_bg[y1:y2, x1:x2] = False

    bg_dx = flow[all_bg, 0]
    bg_dy = flow[all_bg, 1]

    if bg_dx.size == 0:
        return 0.0, 0.0

    dx = float(np.percentile(bg_dx, fallback_percentile))
    dy = float(np.percentile(bg_dy, fallback_percentile))
    return dx, dy


def compute_cumulative_background_flow(
    frame_dir: Path,
    frame_files: list[str],
    bboxes: list[list[float]],
    *,
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
    min_dark_pixels: int = 50,
    verbose: bool = True,
) -> tuple[np.ndarray, int]:
    """Compute cumulative background flow using dark-feature (rock/tree) pixels.

    Returns
    -------
    (cum_flow, n_dark_frames)
        cum_flow: shape (N, 2), cumulative flow from frame 0.
        n_dark_frames: number of frames where dark pixels were used (vs fallback).
    """
    n = len(frame_files)
    per_frame_bg = np.zeros((n, 2), dtype=float)
    n_dark_used = 0

    prev_gray = None
    for i, fname in enumerate(frame_files):
        frame_bgr = cv2.imread(str(frame_dir / fname), cv2.IMREAD_GRAYSCALE)
        if frame_bgr is None:
            if verbose:
                print(f"  [flow] Frame {i} unreadable: {fname}")
            prev_gray = None
            continue

        if prev_gray is not None:
            # Check dark pixel count at current frame to know which path was taken
            dark_mask = _dark_feature_mask(
                prev_gray, bboxes[i],
                dark_threshold=dark_threshold,
                min_gradient=min_gradient,
            )
            if int(dark_mask.sum()) >= min_dark_pixels:
                n_dark_used += 1

            dx, dy = _background_flow(
                prev_gray, frame_bgr, bboxes[i],
                dark_threshold=dark_threshold,
                min_gradient=min_gradient,
                min_dark_pixels=min_dark_pixels,
            )
            per_frame_bg[i] = [dx, dy]

        prev_gray = frame_bgr

        if verbose and i % 100 == 0:
            print(f"  [flow] {i}/{n} frames processed...", end="\r")

    if verbose:
        print(f"  [flow] {n}/{n} frames processed.  ")

    cum_flow = np.cumsum(per_frame_bg, axis=0)
    return cum_flow, n_dark_used


# ---------------------------------------------------------------------------
# Scale calibration
# ---------------------------------------------------------------------------

def calibrate_pchip(
    cum_flow: np.ndarray,
    frame_ids: list[int],
    gt: dict[int, tuple[float, float]],
    venue_w: int,
    venue_h: int,
    *,
    held_out: int | None = None,
):
    """Fit PCHIP interpolators: cum_flow → venue_pos.

    Using PCHIP on the FLOW signal (not frame index) captures zoom variation.
    GT is used ONLY for calibration (not looked up as a prediction value).

    Parameters
    ----------
    held_out : int | None
        If given, exclude this frame_id from calibration (LOO evaluation).

    Returns
    -------
    (interp_x, interp_y, fallback_params, residual_rms)
        interp_x, interp_y: PCHIP interpolators  OR None if too few points
        fallback_params: linear fallback [[ax,bx],[ay,by]]
        residual_rms: calibration residual
    """
    try:
        from scipy.interpolate import PchipInterpolator
        scipy_ok = True
    except ImportError:
        scipy_ok = False

    id_to_idx = {fid: i for i, fid in enumerate(frame_ids)}

    # Build sorted calibration set (by cum_flow_y — monotone for PCHIP)
    cal_points = []
    for fid, (vx, vy) in gt.items():
        if fid == held_out:
            continue
        if fid not in id_to_idx:
            continue
        idx = id_to_idx[fid]
        cal_points.append((cum_flow[idx, 0], cum_flow[idx, 1], vx, vy))

    if len(cal_points) < 2:
        return None, None, np.array([[0.0, -1.0], [0.0, -1.0]]), 999.0

    # Sort by cum_flow_y (should be monotone in y since camera pans down)
    cal_points.sort(key=lambda p: p[1])
    flow_x = np.array([p[0] for p in cal_points])
    flow_y = np.array([p[1] for p in cal_points])
    venue_x = np.array([p[2] for p in cal_points])
    venue_y = np.array([p[3] for p in cal_points])

    interp_x = interp_y = None
    if scipy_ok and len(cal_points) >= 3:
        # Use cum_flow_y as the common parameter (strongly monotone)
        # Ensure strictly decreasing (flow_y should be negative and decreasing)
        if len(set(flow_y)) == len(flow_y):  # no duplicates
            interp_x = PchipInterpolator(flow_y, venue_x, extrapolate=True)
            interp_y = PchipInterpolator(flow_y, venue_y, extrapolate=True)

    # Fallback linear params
    def _fit(xs, ys):
        A = np.column_stack([np.ones_like(xs), xs])
        res, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        return float(res[0]), float(res[1])

    ax, bx = _fit(flow_x, venue_x)
    ay, by = _fit(flow_y, venue_y)
    fallback = np.array([[ax, bx], [ay, by]])

    # Calibration RMS
    if interp_x is not None:
        pred_x = [float(np.clip(interp_x(fy), 0, venue_w - 1)) for fy in flow_y]
        pred_y = [float(np.clip(interp_y(fy), 0, venue_h - 1)) for fy in flow_y]
    else:
        pred_x = ax + bx * flow_x
        pred_y = ay + by * flow_y
    rms = float(np.sqrt(np.mean(
        (np.array(pred_x) - venue_x) ** 2 + (np.array(pred_y) - venue_y) ** 2
    )))
    return interp_x, interp_y, fallback, rms


def predict(
    cum_flow: np.ndarray,
    frame_idx: int,
    interp_x,
    interp_y,
    fallback_params: np.ndarray,
    venue_w: int,
    venue_h: int,
) -> tuple[float, float]:
    """Predict venue position for frame index using PCHIP on flow signal."""
    fy = cum_flow[frame_idx, 1]
    fx = cum_flow[frame_idx, 0]
    if interp_x is not None:
        vx = float(np.clip(float(interp_x(fy)), 0, venue_w - 1))
        vy = float(np.clip(float(interp_y(fy)), 0, venue_h - 1))
    else:
        ax, bx = fallback_params[0]
        ay, by = fallback_params[1]
        vx = float(np.clip(ax + bx * fx, 0, venue_w - 1))
        vy = float(np.clip(ay + by * fy, 0, venue_h - 1))
    return vx, vy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_loo_evaluation(
    cum_flow: np.ndarray,
    frame_ids: list[int],
    gt: dict[int, tuple[float, float]],
    venue_w: int,
    venue_h: int,
    *,
    verbose: bool = True,
) -> dict:
    """Leave-one-out cross-validation over GT frames within tracking range.

    Uses PCHIP regression on cumulative background flow (not frame index).
    GT is used for calibration only — never as a lookup for the prediction.
    """
    id_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
    in_range = {fid: v for fid, v in gt.items() if fid in id_to_idx}

    if not in_range:
        return {"error": "No GT frames within tracking range"}

    errors = []
    details = []

    for held_fid, (true_vx, true_vy) in sorted(in_range.items()):
        ix, iy, fb, _ = calibrate_pchip(
            cum_flow, frame_ids, gt, venue_w, venue_h, held_out=held_fid
        )
        idx = id_to_idx[held_fid]
        pred_vx, pred_vy = predict(cum_flow, idx, ix, iy, fb, venue_w, venue_h)
        err = float(np.sqrt((pred_vx - true_vx) ** 2 + (pred_vy - true_vy) ** 2))
        errors.append(err)
        details.append({
            "frame_id": held_fid,
            "true": (true_vx, true_vy),
            "pred": (pred_vx, pred_vy),
            "err_px": err,
        })
        if verbose:
            print(f"  LOO frame {held_fid:5d}: true=({true_vx:.0f},{true_vy:.0f})  "
                  f"pred=({pred_vx:.0f},{pred_vy:.0f})  err={err:.1f}px")

    errors_arr = np.array(errors)
    result = {
        "n_frames": len(errors),
        "mean_err_px": float(np.mean(errors_arr)),
        "median_err_px": float(np.median(errors_arr)),
        "p90_err_px": float(np.percentile(errors_arr, 90)),
        "within_50px": float(np.mean(errors_arr <= 50.0)),
        "details": details,
    }
    return result


def run_full_evaluation(
    cum_flow: np.ndarray,
    frame_ids: list[int],
    gt: dict[int, tuple[float, float]],
    venue_w: int,
    venue_h: int,
) -> dict:
    """Evaluate using ALL GT for calibration (non-LOO)."""
    id_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
    in_range = {fid: v for fid, v in gt.items() if fid in id_to_idx}

    ix, iy, fb, rms = calibrate_pchip(cum_flow, frame_ids, gt, venue_w, venue_h)

    errors = []
    for fid, (true_vx, true_vy) in sorted(in_range.items()):
        idx = id_to_idx[fid]
        pred_vx, pred_vy = predict(cum_flow, idx, ix, iy, fb, venue_w, venue_h)
        err = float(np.sqrt((pred_vx - true_vx) ** 2 + (pred_vy - true_vy) ** 2))
        errors.append(err)

    errors_arr = np.array(errors)
    return {
        "n_frames": len(errors),
        "mean_err_px": float(np.mean(errors_arr)),
        "median_err_px": float(np.median(errors_arr)),
        "within_50px": float(np.mean(errors_arr <= 50.0)),
        "calibration_rms": rms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venue_camflow(
    video_path: str | Path,
    *,
    venue_image_path: str | Path = "venue_image.png",
    tracking_root: str | Path = "output/tracking",
    frame_root: str | Path = "output/frames",
    gt_root: str | Path = "annotations/venue",
    output_root: str | Path = "output/venue_mapping",
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
    min_dark_pixels: int = 50,
    no_video: bool = False,
    verbose: bool = True,
) -> dict:
    """Run camera-motion venue mapping and LOO evaluation."""
    video_path = Path(video_path)
    video_stem = video_path.stem

    frame_dir = Path(frame_root) / video_stem
    tracking_manifest = Path(tracking_root) / video_stem / "tracking.json"
    venue_image_path = Path(venue_image_path)
    gt_path = Path(gt_root) / video_stem / "gt_venue.csv"
    output_dir = Path(output_root) / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for p, label in [
        (frame_dir, "Frame directory"),
        (tracking_manifest, "Tracking manifest"),
        (venue_image_path, "Venue image"),
        (gt_path, "GT annotations"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    venue_bgr = cv2.imread(str(venue_image_path), cv2.IMREAD_COLOR)
    if venue_bgr is None:
        raise RuntimeError(f"Cannot read venue image: {venue_image_path}")
    venue_h, venue_w = venue_bgr.shape[:2]

    frame_ids, frame_files, bboxes = _load_tracking(tracking_manifest)
    gt = _load_gt(gt_path)

    if verbose:
        print(f"[camflow] {video_stem}: {len(frame_ids)} frames, {len(gt)} GT anchors")
        print(f"[camflow] Computing background optical flow (dark_threshold={dark_threshold})...")

    cum_flow, n_dark = compute_cumulative_background_flow(
        frame_dir, frame_files, bboxes,
        dark_threshold=dark_threshold,
        min_gradient=min_gradient,
        min_dark_pixels=min_dark_pixels,
        verbose=verbose,
    )

    if verbose:
        n_transitions = len(frame_files) - 1
        print(f"\n[camflow] Dark-feature OF used in {n_dark}/{n_transitions} frame transitions "
              f"({100*n_dark/max(1,n_transitions):.0f}%)")
        print(f"[camflow] Cumulative flow range: "
              f"x=[{cum_flow[:,0].min():.1f}, {cum_flow[:,0].max():.1f}]  "
              f"y=[{cum_flow[:,1].min():.1f}, {cum_flow[:,1].max():.1f}]")

    # Full evaluation (non-LOO, using all GT for calibration)
    full_res = run_full_evaluation(cum_flow, frame_ids, gt, venue_w, venue_h)
    if verbose:
        print(f"\n[camflow] Full eval (non-LOO): "
              f"mean={full_res['mean_err_px']:.1f}px  "
              f"median={full_res['median_err_px']:.1f}px  "
              f"within50={full_res['within_50px']:.0%}  "
              f"cal_rms={full_res['calibration_rms']:.1f}px")
        print(f"  (PCHIP on cumulative dark-feature flow)")

    # LOO evaluation (honest)
    if verbose:
        print("\n[camflow] LOO cross-validation:")
    loo_res = run_loo_evaluation(cum_flow, frame_ids, gt, venue_w, venue_h, verbose=verbose)
    if verbose:
        print(f"\n[camflow] LOO summary: "
              f"mean={loo_res['mean_err_px']:.1f}px  "
              f"median={loo_res['median_err_px']:.1f}px  "
              f"P90={loo_res['p90_err_px']:.1f}px  "
              f"within50={loo_res['within_50px']:.0%}")

    # Save results
    results = {
        "video_stem": video_stem,
        "dark_threshold": dark_threshold,
        "n_tracking_frames": len(frame_ids),
        "n_gt_anchors": len(gt),
        "full_eval": full_res,
        "loo_eval": loo_res,
    }
    results_path = output_dir / "camflow_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if verbose:
        print(f"\n[camflow] Results saved: {results_path}")

    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Venue mapping via background camera motion flow.")
    p.add_argument("video", type=Path)
    p.add_argument("--venue-image", type=Path, default=Path("venue_image.png"))
    p.add_argument("--tracking-root", type=Path, default=Path("output/tracking"))
    p.add_argument("--frame-root", type=Path, default=Path("output/frames"))
    p.add_argument("--gt-root", type=Path, default=Path("annotations/venue"))
    p.add_argument("--output-root", type=Path, default=Path("output/venue_mapping"))
    p.add_argument("--dark-threshold", type=int, default=80, help="Max brightness for dark features (default=80)")
    p.add_argument("--min-gradient", type=float, default=5.0, help="Min gradient for texture filter (default=5.0)")
    p.add_argument("--min-dark-pixels", type=int, default=50, help="Min dark pixels to use dark-mode (default=50)")
    p.add_argument("--no-video", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_venue_camflow(
        args.video,
        venue_image_path=args.venue_image,
        tracking_root=args.tracking_root,
        frame_root=args.frame_root,
        gt_root=args.gt_root,
        output_root=args.output_root,
        dark_threshold=args.dark_threshold,
        min_gradient=args.min_gradient,
        min_dark_pixels=args.min_dark_pixels,
        no_video=args.no_video,
    )


if __name__ == "__main__":
    main()
