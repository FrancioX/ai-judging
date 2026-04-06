"""Venue image registration via multi-scale dark-patch template matching.

**Problem**
-----------
The broadcast camera is ~10x zoomed into the venue image, which puts it
far outside SIFT/ORB scale range (~5-6x).  Template matching at the right
scale is the natural fit: the video frame IS a scaled crop of the venue
image, and we need to find where that crop falls.

**Algorithm**
-------------
1. Compute gradient magnitude of dark-feature (rocks/trees) pixels in the
   full venue image — this is the search space.
2. For each candidate video frame (every ``--frame-step`` frames):
   a. Skip if the frame has fewer than ``--min-dark-px-frame`` dark-feature
      pixels.  Early snowy frames (< 1% dark) produce false positives; rocky
      mid/late frames (> 3-5%) give reliable matches.  Default: 50 000.
   b. Compute gradient magnitude of dark-feature pixels in the frame.
   c. Sweep over a range of candidate scales (``--scale-min`` to
      ``--scale-max``).  At each scale, resize the frame gradient to
      ``scale × frame_size`` and run ``cv2.TM_CCOEFF_NORMED`` against the
      venue gradient.
   d. Record the best NCC score + match location across all scales.
3. From the filtered matches, add temporal-consistency filtering: for each
   pair (early frame i, late frame j), the matched venue_y should be
   monotone increasing (early < late, since the athlete goes top→bottom).
   Rank matches by how many pairs they participate in consistent ordering.
4. Use the most-consistent, highest-NCC match as the **anchor frame**.
5. Project the athlete's centre (from tracking) into venue coordinates at
   the anchor frame scale.
6. If multiple reliable matches are available, fit a PCHIP curve
   (cumulative-flow → venue position) using all filtered matches as
   pseudo-GT calibration points.  This is more robust than a single linear
   scale derived from one anchor.
7. Evaluate against GT annotations via standard error metrics.

**Why early frames produce false positives**
--------------------------------------------
The top of the venue image (~y < 200) is snow-covered (≈1.2% dark pixels).
Even though a video frame from the start of the run may show some dark
features (rocks at slope edges or foreground), the NCC maximises at a
different part of the venue (the dark rock bands in the middle/lower
section).  Filtering on ``min_dark_px_frame`` discards these early frames.

**Sky/mountain-edge matching for early frames**
-----------------------------------------------
Frames at the start of the run show the upper slope with sky visible above
the mountain silhouette.  A second matching pass (after the dark-feature
scan) runs these early frames against a venue gradient restricted to the
top ``max_sky_venue_y`` rows.  The zero-padding below that row naturally
prevents matches in the rock-band zone.  The best sky-edge match is added
as an early-flow calibration point for PCHIP, bridging the gap between the
first frame (flow≈0) and the dark-feature anchor (flow≈-440).

**Verification outputs** (written to ``output/venue_registration/<stem>/``)
---------------------------------------------------------------------------
- ``venue_dark_grad.jpg``  — venue gradient restricted to dark features
- ``venue_sky_grad.jpg``   — venue gradient of top zone (sky/mountain edge)
- ``anchor_match.jpg``     — venue with matched rectangle + anchor frame
- ``score_curve.json``     — per-frame best NCC score (check for clear peak)
- ``registration_results.json`` — full per-frame predictions + metrics

**Usage**
---------
::

    uv run python scripts/venue_registration.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4"

    # Lower frame-dark-px threshold if too few matches are found
    uv run python scripts/venue_registration.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4" \\
        --min-dark-px-frame 20000

    # Tune scale range
    uv run python scripts/venue_registration.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4" \\
        --scale-min 0.06 --scale-max 0.18 --n-scales 20
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.utils.optical_flow import dark_feature_mask as _dark_feature_mask


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_tracking(manifest_path: Path) -> tuple[list[int], list[str], list[list[float]]]:
    """Return (frame_ids, frame_files, bboxes).

    frame_ids are the FILE NUMBERS extracted from the filename
    (e.g. frame_000250.jpg → 250).
    """
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


def _load_velocity(velocity_path: Path) -> dict[int, list[float]]:
    """Return {file_number: [cum_flow_dx, cum_flow_dy]} from velocity.json.

    velocity.json stores sequential frame_id (0,1,2...) and frame_file
    (e.g. 'frame_000250.jpg').  We key by the FILE NUMBER (250) to match
    tracking.json which uses the same file number convention.
    Cumulative flow is computed by accumulating per-frame camera_flow_dx/dy.
    """
    with velocity_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    result: dict[int, list[float]] = {}
    cum_dx = 0.0
    cum_dy = 0.0
    for fr in data.get("frames", []):
        fname = fr.get("frame_file", "")
        try:
            file_num = int(
                fname.replace("frame_", "").replace(".jpg", "").replace(".png", "")
            )
        except ValueError:
            continue
        cum_dx += fr.get("camera_flow_dx", 0.0)
        cum_dy += fr.get("camera_flow_dy", 0.0)
        result[file_num] = [cum_dx, cum_dy]
    return result


def _load_gt(gt_path: Path) -> dict[int, tuple[float, float]]:
    gt: dict[int, tuple[float, float]] = {}
    if not gt_path.exists():
        return gt
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
# Venue gradient images (computed once)
# ---------------------------------------------------------------------------

def compute_venue_sky_gradient(
    venue_gray: np.ndarray,
    *,
    max_venue_y: int = 350,
    min_gradient: float = 20.0,
) -> np.ndarray:
    """All-pixel gradient of the venue's top zone (sky/mountain-edge region).

    Unlike dark-feature matching, sky-edge matching needs ALL strong gradients
    in the top rows — the sky-mountain boundary spans both bright (sky) and
    dark (mountain top) pixels.

    The returned array is full venue size but zeros below ``max_venue_y``.
    This naturally restricts ``matchTemplate`` results to the top zone
    (matches below ``max_venue_y - template_h`` will have depressed NCC due
    to overlap with the zero-padded region).

    Parameters
    ----------
    max_venue_y : int
        Only the top ``max_venue_y`` rows are populated.  Default 350 —
        covers the sky and upper mountain silhouette in typical venue images.
    min_gradient : float
        Only strong edges are kept (higher than the 5.0 used for dark
        features) to focus on the sharp sky-mountain boundary.
    """
    vh, vw = venue_gray.shape[:2]
    top = min(max_venue_y, vh)
    region = venue_gray[:top].astype(np.float32)
    gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    texture = grad > min_gradient
    full = np.zeros((vh, vw), dtype=np.float32)
    full[:top] = grad * texture.astype(np.float32)
    return full


def compute_venue_dark_gradient(
    venue_gray: np.ndarray,
    *,
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
) -> np.ndarray:
    """Gradient magnitude of dark-feature pixels in the venue image.

    Snow / sky are bright → zero in the output.  Rocks and trees are dark +
    textured → high gradient → present in the output.
    """
    dark_mask = venue_gray < dark_threshold
    gx = cv2.Sobel(venue_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(venue_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    texture_mask = grad > min_gradient
    combined = dark_mask & texture_mask
    return (grad * combined.astype(np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-frame template matching
# ---------------------------------------------------------------------------

def _frame_dark_gradient(
    gray: np.ndarray,
    bbox: list[float],
    *,
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
) -> np.ndarray:
    """Gradient magnitude of dark-feature pixels, athlete bbox excluded."""
    mask = _dark_feature_mask(
        gray, bbox, dark_threshold=dark_threshold, min_gradient=min_gradient
    )
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    return (grad * mask.astype(np.float32)).astype(np.float32)


def _frame_all_gradient(
    gray: np.ndarray,
    bbox: list[float],
    *,
    min_gradient: float = 20.0,
) -> np.ndarray:
    """All-pixel strong gradient of the frame, athlete bbox excluded.

    Used for sky/mountain-edge matching in early frames where dark-feature
    density is too low.  Higher ``min_gradient`` threshold selects only sharp
    edges (like the sky-mountain boundary) rather than subtle textures.
    """
    fh, fw = gray.shape[:2]
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    texture = grad > min_gradient
    # Exclude athlete bbox
    bx1, by1, bx2, by2 = [int(v) for v in bbox[:4]]
    texture[max(0, by1):min(fh, by2), max(0, bx1):min(fw, bx2)] = False
    return (grad * texture.astype(np.float32)).astype(np.float32)


def _match_frame_at_scale(
    frame_grad: np.ndarray,
    venue_grad: np.ndarray,
    scale: float,
) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """Match scaled frame gradient against venue gradient.

    Returns
    -------
    (ncc_score, top_left_venue, template_size)
    """
    fh, fw = frame_grad.shape[:2]
    th = max(4, int(round(fh * scale)))
    tw = max(4, int(round(fw * scale)))

    vh, vw = venue_grad.shape[:2]
    if th >= vh or tw >= vw:
        return -2.0, (0, 0), (tw, th)

    template = cv2.resize(frame_grad, (tw, th), interpolation=cv2.INTER_AREA)
    if template.sum() < 1.0:
        return -2.0, (0, 0), (tw, th)

    result = cv2.matchTemplate(venue_grad, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return float(max_val), (int(max_loc[0]), int(max_loc[1])), (tw, th)


def match_frame_to_venue(
    frame_grad: np.ndarray,
    venue_grad: np.ndarray,
    scales: np.ndarray,
) -> dict:
    """Sweep scales and return best match info."""
    best: dict = {
        "score": -999.0, "scale": float(scales[0]),
        "venue_x": 0, "venue_y": 0, "template_w": 0, "template_h": 0,
    }
    for scale in scales:
        score, (tx, ty), (tw, th) = _match_frame_at_scale(frame_grad, venue_grad, float(scale))
        if score > best["score"]:
            best = {
                "score": score, "scale": float(scale),
                "venue_x": tx, "venue_y": ty,
                "template_w": tw, "template_h": th,
            }
    return best


# ---------------------------------------------------------------------------
# Athlete venue position from match
# ---------------------------------------------------------------------------

def _athlete_venue_pos(
    match: dict,
    bbox: list[float],
) -> tuple[float, float]:
    """Project athlete centre from frame coords to venue coords."""
    bx1, by1, bx2, by2 = bbox[:4]
    frame_cx = (bx1 + bx2) / 2.0
    frame_cy = (by1 + by2) / 2.0
    vx = match["venue_x"] + frame_cx * match["scale"]
    vy = match["venue_y"] + frame_cy * match["scale"]
    return vx, vy


# ---------------------------------------------------------------------------
# Temporal-consistency anchor selection
# ---------------------------------------------------------------------------

def _temporal_consistency_scores(match_results: list[dict]) -> dict[int, int]:
    """Compute temporal-consistency score for each match.

    For each pair (m_i, m_j) with fid_i < fid_j, a correct match should
    have athlete_vy_i <= athlete_vy_j (athlete goes top→bottom in venue,
    so venue_y increases over time).  Count consistent pairs per match.

    Requires ``athlete_vx`` / ``athlete_vy`` fields in each match dict.
    """
    sorted_matches = sorted(match_results, key=lambda m: m["frame_id"])
    n = len(sorted_matches)
    consistent_count: dict[int, int] = {m["frame_id"]: 0 for m in sorted_matches}

    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = sorted_matches[i], sorted_matches[j]
            if mi["athlete_vy"] <= mj["athlete_vy"]:
                consistent_count[mi["frame_id"]] += 1
                consistent_count[mj["frame_id"]] += 1

    return consistent_count


def _temporal_consistent_anchor(match_results: list[dict]) -> dict:
    """Return best anchor by temporal-consistency (primary) + NCC score (secondary)."""
    counts = _temporal_consistency_scores(match_results)
    return max(match_results, key=lambda m: (counts[m["frame_id"]], m["score"]))


def _filter_consistent_matches(
    match_results: list[dict],
    *,
    min_consistency_frac: float = 0.5,
) -> list[dict]:
    """Return the subset of matches with consistency count ≥ min_frac × max_count.

    This removes temporal outliers (false-positive matches whose venue_y is
    inconsistent with the majority ordering) before PCHIP calibration.
    """
    counts = _temporal_consistency_scores(match_results)
    max_count = max(counts.values()) if counts else 0
    if max_count == 0:
        return match_results
    threshold = int(min_consistency_frac * max_count)
    return [m for m in match_results if counts[m["frame_id"]] >= threshold]


# ---------------------------------------------------------------------------
# PCHIP calibration & prediction (replicates venue_camflow.py approach)
# ---------------------------------------------------------------------------

def _calibrate_pchip(
    cum_flow: dict[int, list[float]],
    frame_ids: list[int],
    pseudo_gt: dict[int, tuple[float, float]],
    venue_w: int,
    venue_h: int,
) -> tuple:
    """Fit PCHIP: cumulative-flow_y → venue position.

    Parameters
    ----------
    pseudo_gt : dict[int, (vx, vy)]
        Calibration points (frame_id → venue athlete position).

    Returns
    -------
    (interp_x, interp_y, fallback_params, flow_y_range)
        flow_y_range : (min_fy, max_fy) — calibration range; PCHIP is only
        valid inside this interval; the fallback linear model is used outside.
    """
    try:
        from scipy.interpolate import PchipInterpolator
        scipy_ok = True
    except ImportError:
        scipy_ok = False

    cal_points = []
    for fid, (vx, vy) in pseudo_gt.items():
        cf = cum_flow.get(fid)
        if cf is None:
            continue
        cal_points.append((cf[0], cf[1], vx, vy))

    if len(cal_points) < 2:
        return None, None, np.array([[0.0, 0.0], [0.0, 0.0]]), (0.0, 0.0)

    cal_points.sort(key=lambda p: p[1])  # sort by cum_flow_y
    flow_x = np.array([p[0] for p in cal_points])
    flow_y = np.array([p[1] for p in cal_points])
    venue_x = np.array([p[2] for p in cal_points])
    venue_y = np.array([p[3] for p in cal_points])

    fy_range = (float(flow_y.min()), float(flow_y.max()))

    interp_x = interp_y = None
    if scipy_ok and len(cal_points) >= 3 and len(set(flow_y)) == len(flow_y):
        # extrapolate=False → NaN outside range → will fall back to linear
        interp_x = PchipInterpolator(flow_y, venue_x, extrapolate=False)
        interp_y = PchipInterpolator(flow_y, venue_y, extrapolate=False)

    def _fit(xs, ys):
        A = np.column_stack([np.ones_like(xs), xs])
        res, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        return float(res[0]), float(res[1])

    ax, bx = _fit(flow_x, venue_x)
    ay, by_ = _fit(flow_y, venue_y)
    fallback = np.array([[ax, bx], [ay, by_]])
    return interp_x, interp_y, fallback, fy_range


def _predict_venue(
    cum_flow: dict[int, list[float]],
    fid: int,
    interp_x,
    interp_y,
    fallback: np.ndarray,
    venue_w: int,
    venue_h: int,
    fy_range: tuple[float, float] = (-1e9, 1e9),
) -> tuple[float, float] | None:
    """Predict venue position.

    Uses PCHIP within the calibration flow-y range; linear fallback outside.
    """
    cf = cum_flow.get(fid)
    if cf is None:
        return None
    fy, fx = cf[1], cf[0]
    in_range = fy_range[0] <= fy <= fy_range[1]

    if interp_x is not None and in_range:
        px = float(interp_x(fy))
        py = float(interp_y(fy))
        if not (np.isnan(px) or np.isnan(py)):
            vx = float(np.clip(px, 0, venue_w - 1))
            vy = float(np.clip(py, 0, venue_h - 1))
            return vx, vy

    # Fallback: linear regression on full calibration set
    ax, bx = fallback[0]
    ay, by_ = fallback[1]
    vx = float(np.clip(ax + bx * fx, 0, venue_w - 1))
    vy = float(np.clip(ay + by_ * fy, 0, venue_h - 1))
    return vx, vy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    positions: dict[int, tuple[float, float]],
    gt: dict[int, tuple[float, float]],
    *,
    verbose: bool = True,
) -> dict:
    """Compare predicted positions vs GT."""
    errors = []
    details = []
    for fid in sorted(gt):
        if fid not in positions:
            continue
        true_vx, true_vy = gt[fid]
        pred_vx, pred_vy = positions[fid]
        err = float(np.sqrt((pred_vx - true_vx) ** 2 + (pred_vy - true_vy) ** 2))
        errors.append(err)
        details.append({
            "frame_id": fid,
            "true": [true_vx, true_vy],
            "pred": [pred_vx, pred_vy],
            "err_px": err,
        })
        if verbose:
            print(f"  frame {fid:5d}: true=({true_vx:.0f},{true_vy:.0f})  "
                  f"pred=({pred_vx:.0f},{pred_vy:.0f})  err={err:.1f}px")

    if not errors:
        return {"error": "No GT frames in tracking range"}

    arr = np.array(errors)
    return {
        "n_frames": len(errors),
        "mean_err_px": float(np.mean(arr)),
        "median_err_px": float(np.median(arr)),
        "p90_err_px": float(np.percentile(arr, 90)),
        "within_50px": float(np.mean(arr <= 50.0)),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _save_venue_dark_grad_jpg(venue_grad: np.ndarray, out_path: Path) -> None:
    norm = cv2.normalize(venue_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(out_path), norm)


def _save_anchor_match_jpg(
    venue_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    match: dict,
    athlete_venue: tuple[float, float],
    out_path: Path,
) -> None:
    """Save venue image with matched rectangle + anchor frame side-by-side."""
    venue_vis = venue_bgr.copy()
    x0, y0 = match["venue_x"], match["venue_y"]
    tw, th = match["template_w"], match["template_h"]
    cv2.rectangle(venue_vis, (x0, y0), (x0 + tw, y0 + th), (0, 255, 0), 3)
    ax, ay = int(round(athlete_venue[0])), int(round(athlete_venue[1]))
    cv2.circle(venue_vis, (ax, ay), 10, (0, 0, 255), -1)
    cv2.putText(
        venue_vis,
        f"NCC={match['score']:.3f}  scale={match['scale']:.3f}",
        (x0, max(15, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )
    vh, vw = venue_vis.shape[:2]
    fh, fw = frame_bgr.shape[:2]
    target_h = max(vh, fh)
    venue_panel = cv2.resize(
        venue_vis, (int(vw * target_h / vh), target_h), interpolation=cv2.INTER_AREA
    )
    frame_panel = cv2.resize(
        frame_bgr, (int(fw * target_h / fh), target_h), interpolation=cv2.INTER_AREA
    )
    cv2.imwrite(str(out_path), np.concatenate([venue_panel, frame_panel], axis=1))


# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------

def _write_venue_video(
    frame_dir: Path,
    frame_files: list[str],
    frame_ids: list[int],
    venue_bgr: np.ndarray,
    all_positions: dict[int, tuple[float, float]],
    output_path: Path,
    *,
    gt: dict[int, tuple[float, float]] | None = None,
    start_anchor_region: tuple[int, int, int, int] | None = None,
    fps: float = 30.0,
    trail_length: int = 80,
    verbose: bool = True,
) -> None:
    """Render side-by-side: original frame (left) + venue with predicted dot (right).

    GT points are drawn as yellow crosses.  The annotated start_anchor_region
    is drawn as a white rectangle if provided.
    """
    from collections import deque
    venue_h, venue_w = venue_bgr.shape[:2]
    trail: deque[tuple[int, int]] = deque(maxlen=trail_length)

    writer = None
    frame_w = frame_h = None

    for i, (fname, fid) in enumerate(zip(frame_files, frame_ids)):
        frame_bgr = cv2.imread(str(frame_dir / fname), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        if frame_w is None:
            frame_h, frame_w = frame_bgr.shape[:2]
            scale = frame_h / venue_h
            panel_w = int(venue_w * scale)
            panel_h = frame_h
            out_w = frame_w + panel_w
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, frame_h))

        venue_panel = venue_bgr.copy()

        # Draw annotated start region
        if start_anchor_region is not None:
            ax1, ay1, ax2, ay2 = start_anchor_region
            cv2.rectangle(venue_panel, (ax1, ay1), (ax2, ay2), (200, 200, 200), 2)

        # Draw GT crosses
        if gt:
            for gt_fid, (gt_vx, gt_vy) in gt.items():
                cv2.drawMarker(venue_panel, (int(gt_vx), int(gt_vy)),
                               color=(0, 255, 255), markerType=cv2.MARKER_CROSS,
                               markerSize=14, thickness=2)

        # Update trail and draw
        if fid in all_positions:
            vx, vy = all_positions[fid]
            trail.append((int(round(vx)), int(round(vy))))

        trail_pts = list(trail)
        if len(trail_pts) >= 2:
            n_seg = len(trail_pts) - 1
            for j in range(n_seg):
                alpha = (j + 1) / max(1, n_seg)
                color = (int(20 + 120 * alpha), int(40 + 180 * alpha), int(200 + 30 * alpha))
                thickness = 1 if alpha < 0.4 else 2
                cv2.line(venue_panel, trail_pts[j], trail_pts[j + 1], color, thickness,
                         lineType=cv2.LINE_AA)

        if trail_pts:
            cur = trail_pts[-1]
            cv2.circle(venue_panel, cur, radius=8, color=(20, 40, 240), thickness=-1,
                       lineType=cv2.LINE_AA)
            cv2.circle(venue_panel, cur, radius=12, color=(180, 220, 255), thickness=2,
                       lineType=cv2.LINE_AA)

        venue_small = cv2.resize(venue_panel, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        combined = np.concatenate([frame_bgr, venue_small], axis=1)
        writer.write(combined)

        if verbose and i % 200 == 0:
            print(f"  [video] {i}/{len(frame_files)} frames...", end="\r")

    if writer is not None:
        writer.release()
    if verbose:
        n = len(frame_files)
        print(f"  [video] {n}/{n} frames done. → {output_path}")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_venue_registration(
    video_path: str | Path,
    *,
    venue_image_path: str | Path = "venue_image.png",
    tracking_root: str | Path = "output/tracking",
    velocity_root: str | Path = "output/velocity",
    frame_root: str | Path = "output/frames",
    gt_root: str | Path = "annotations/venue",
    output_root: str | Path = "output/venue_registration",
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
    scale_min: float = 0.08,
    scale_max: float = 0.15,
    n_scales: int = 15,
    frame_step: int = 10,
    min_dark_px_frame: int = 50_000,
    start_anchor_region: tuple[int, int, int, int] | None = None,
    search_region: tuple[int, int, int, int] | None = None,
    no_video: bool = False,
    verbose: bool = True,
) -> dict:
    """Run dark-patch template matching venue registration.

    Parameters
    ----------
    min_dark_px_frame : int
        Minimum number of dark-feature pixels in a VIDEO FRAME for it to be
        used as a matching candidate.  Early snowy frames have ≈10-25 K dark
        pixels and produce false positives; frames over rocky terrain have
        ≫50 K.  Default 50 000.  Lower to 20 000 if few matches are found.
    start_anchor_region : (x1, y1, x2, y2) | None
        Small venue rectangle marking the approximate athlete start area
        (annotated once with ``scripts/annotate_venue_start.py``).  When set,
        the sky/edge pass for early frames is restricted to this rectangle,
        ensuring the start anchor lands in the correct venue zone.  The
        full-venue dark-feature scan proceeds without this restriction, so
        late-run matches are unaffected.

        Auto-loaded from ``annotations/venue/start_region.json`` when that
        file exists and ``start_anchor_region`` is not explicitly provided.
    search_region : (x1, y1, x2, y2) | None
        Restrict ALL template matching (both dark-feature and sky-edge) to
        this sub-region of the venue.  Useful when you know the full run
        corridor (e.g. ``--search-region 900,0,1500,1440``).  Unlike
        ``start_anchor_region``, this constrains every frame, not just the
        anchor.  Overrides ``start_anchor_region`` for the sky-edge pass if
        both are set.
    """
    video_path = Path(video_path)
    stem = video_path.stem

    frame_dir = Path(frame_root) / stem
    tracking_manifest = Path(tracking_root) / stem / "tracking.json"
    velocity_json = Path(velocity_root) / stem / "velocity.json"
    venue_img_path = Path(venue_image_path)
    gt_path = Path(gt_root) / stem / "gt_venue.csv"
    out_dir = Path(output_root) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for p, label in [
        (frame_dir, "Frame directory"),
        (tracking_manifest, "Tracking manifest"),
        (venue_img_path, "Venue image"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    # Load venue image
    venue_bgr = cv2.imread(str(venue_img_path), cv2.IMREAD_COLOR)
    if venue_bgr is None:
        raise RuntimeError(f"Cannot read venue image: {venue_img_path}")
    venue_gray = cv2.cvtColor(venue_bgr, cv2.COLOR_BGR2GRAY)
    venue_h, venue_w = venue_gray.shape[:2]

    # Load tracking
    frame_ids, frame_files, bboxes = _load_tracking(tracking_manifest)
    n_frames = len(frame_ids)

    # Load cumulative flow
    cum_flow_by_fid: dict[int, list[float]] = {}
    if velocity_json.exists():
        cum_flow_by_fid = _load_velocity(velocity_json)
        if verbose:
            print(f"[reg] Loaded cumulative flow for {len(cum_flow_by_fid)} frames")
    else:
        if verbose:
            print(f"[reg] velocity.json not found — PCHIP chaining disabled")

    gt = _load_gt(gt_path)
    if verbose:
        print(f"[reg] {stem}: {n_frames} tracking frames, {len(gt)} GT anchors")

    # Step 1: Venue dark gradient
    if verbose:
        print(f"[reg] Computing venue dark-feature gradient (threshold={dark_threshold})...")
    venue_grad = compute_venue_dark_gradient(
        venue_gray, dark_threshold=dark_threshold, min_gradient=min_gradient
    )
    venue_dark_frac = float((venue_grad > 0).sum()) / venue_grad.size
    if verbose:
        print(f"[reg] Venue dark-feature coverage: {venue_dark_frac:.1%}")
    _save_venue_dark_grad_jpg(venue_grad, out_dir / "venue_dark_grad.jpg")

    # Auto-load start_anchor_region from annotation file if not provided.
    if start_anchor_region is None:
        _sr_path = Path("annotations/venue/start_region.json")
        if _sr_path.exists():
            try:
                _sr = json.loads(_sr_path.read_text())
                start_anchor_region = (_sr["x1"], _sr["y1"], _sr["x2"], _sr["y2"])
                if verbose:
                    print(f"[reg] Loaded start anchor region from {_sr_path}: "
                          f"{start_anchor_region}")
            except Exception as exc:
                if verbose:
                    print(f"[reg] Warning: could not load start_region.json: {exc}")

    # search_region (CLI --search-region) crops ALL matching.
    # Coordinates stored as venue-space offset so all match results stay
    # in full venue coordinates regardless of cropping.
    vx_offset, vy_offset = 0, 0
    if search_region is not None:
        sx1, sy1, sx2, sy2 = search_region
        sx1 = max(0, min(sx1, venue_w - 1))
        sy1 = max(0, min(sy1, venue_h - 1))
        sx2 = max(sx1 + 1, min(sx2, venue_w))
        sy2 = max(sy1 + 1, min(sy2, venue_h))
        venue_grad = venue_grad[sy1:sy2, sx1:sx2]
        vx_offset, vy_offset = sx1, sy1
        if verbose:
            print(f"[reg] Search region restricted to venue ({sx1},{sy1})–({sx2},{sy2}) "
                  f"({sx2-sx1}×{sy2-sy1}px); offset will be added to match coords.")

    # Step 2: Per-frame template matching
    scales = np.linspace(scale_min, scale_max, n_scales)
    if verbose:
        print(f"[reg] Scanning {n_frames} frames (step={frame_step}), {n_scales} scales "
              f"[{scale_min:.3f}–{scale_max:.3f}], min_dark_px_frame={min_dark_px_frame}...")

    score_curve: list[dict] = []
    match_results: list[dict] = []
    early_frames: list[tuple] = []  # (idx, fid, fname, bbox, gray) for sky-edge pass

    for idx in range(0, n_frames, frame_step):
        fid = frame_ids[idx]
        fname = frame_files[idx]
        bbox = bboxes[idx]

        frame_gray = cv2.imread(str(frame_dir / fname), cv2.IMREAD_GRAYSCALE)
        if frame_gray is None:
            continue

        frame_grad = _frame_dark_gradient(
            frame_gray, bbox, dark_threshold=dark_threshold, min_gradient=min_gradient
        )
        dark_px = int((frame_grad > 0).sum())

        if dark_px < min_dark_px_frame:
            score_curve.append({"frame_id": fid, "score": -1.0, "dark_px": dark_px,
                                 "filtered": True})
            # Store for sky-edge second pass
            early_frames.append((idx, fid, fname, bbox, frame_gray))
            continue

        match = match_frame_to_venue(frame_grad, venue_grad, scales)
        match["venue_x"] += vx_offset
        match["venue_y"] += vy_offset
        match["frame_id"] = fid
        match["frame_idx"] = idx
        match["dark_px"] = dark_px
        # Annotate athlete venue position immediately
        vx, vy = _athlete_venue_pos(match, bbox)
        match["athlete_vx"] = vx
        match["athlete_vy"] = vy
        match_results.append(match)
        score_curve.append({"frame_id": fid, "score": match["score"], "dark_px": dark_px,
                             "filtered": False, "athlete_vy": vy})

        if verbose and len(match_results) % 5 == 0:
            print(f"  [reg] {idx}/{n_frames} frames, {len(match_results)} valid matches...",
                  end="\r")

    if verbose:
        print()

    (out_dir / "score_curve.json").write_text(
        json.dumps(score_curve, indent=2), encoding="utf-8"
    )

    if not match_results:
        raise RuntimeError(
            f"No frames passed the dark-pixel filter (min_dark_px_frame={min_dark_px_frame}). "
            "Try lowering --min-dark-px-frame."
        )

    n_filtered = sum(1 for s in score_curve if s.get("filtered"))
    if verbose:
        print(f"[reg] {len(match_results)} frames passed dark filter "
              f"({n_filtered} early frames → sky-edge pass)")

    # Sky/start-edge pass: match early frames against the venue top zone.
    # If start_anchor_region is set (from annotations/venue/start_region.json),
    # restrict the venue search to that rectangle — this forces the early-frame
    # anchor into the correct venue zone regardless of where other dark features
    # happen to score higher.  The full-venue dark scan above is unaffected.
    # If only search_region (CLI corridor) is set, apply that same crop here.
    max_sky_venue_y = 350
    sky_match_results: list[dict] = []
    if early_frames:
        venue_sky_grad = compute_venue_sky_gradient(
            venue_gray, max_venue_y=max_sky_venue_y, min_gradient=20.0
        )

        # Determine the x-range to use for sky/anchor matching.
        # Priority: start_anchor_region (from JSON) > search_region (CLI) > full venue.
        sky_offset_x, sky_offset_y = 0, 0
        if start_anchor_region is not None:
            ax1, ay1, ax2, ay2 = start_anchor_region
            ax1 = max(0, min(ax1, venue_w - 1))
            ay1 = max(0, min(ay1, venue_h - 1))
            ax2 = max(ax1 + 1, min(ax2, venue_w))
            ay2 = max(ay1 + 1, min(ay2, venue_h))
            venue_sky_grad = venue_sky_grad[ay1:ay2, ax1:ax2]
            sky_offset_x, sky_offset_y = ax1, ay1
            if verbose:
                print(f"[reg] Sky/anchor pass: matching {len(early_frames)} early frames "
                      f"restricted to start_anchor_region ({ax1},{ay1})–({ax2},{ay2})...")
        elif search_region is not None:
            venue_sky_grad = venue_sky_grad[vy_offset:vy_offset + venue_grad.shape[0],
                                            vx_offset:vx_offset + venue_grad.shape[1]]
            sky_offset_x, sky_offset_y = vx_offset, vy_offset
            if verbose:
                print(f"[reg] Sky-edge pass: matching {len(early_frames)} early frames "
                      f"against venue top {max_sky_venue_y}px (search_region applied)...")
        else:
            if verbose:
                print(f"[reg] Sky-edge pass: matching {len(early_frames)} early frames "
                      f"against full venue top {max_sky_venue_y}px...")

        _save_venue_dark_grad_jpg(venue_sky_grad, out_dir / "venue_sky_grad.jpg")
        for idx, fid, fname, bbox, frame_gray in early_frames:
            frame_all_grad = _frame_all_gradient(frame_gray, bbox, min_gradient=20.0)
            sky_match = match_frame_to_venue(frame_all_grad, venue_sky_grad, scales)
            if sky_match["score"] < 0.0:
                continue
            sky_match["venue_x"] += sky_offset_x
            sky_match["venue_y"] += sky_offset_y
            sky_match["frame_id"] = fid
            sky_match["frame_idx"] = idx
            sky_match["dark_px"] = int((frame_all_grad > 0).sum())
            sky_match["match_type"] = "sky_edge"
            vx, vy = _athlete_venue_pos(sky_match, bbox)
            sky_match["athlete_vx"] = vx
            sky_match["athlete_vy"] = vy
            sky_match_results.append(sky_match)
        if verbose and sky_match_results:
            best_sky = max(sky_match_results, key=lambda m: m["score"])
            print(f"[reg] Sky/anchor: {len(sky_match_results)} matches, "
                  f"best NCC={best_sky['score']:.3f} at frame {best_sky['frame_id']} "
                  f"→ venue ({best_sky['athlete_vx']:.0f},{best_sky['athlete_vy']:.0f})")

    # Step 3: Temporal-consistency anchor selection
    # Use dark-feature matches for the anchor (sky-edge are added to calibration
    # only, not used as primary anchor since their venue_y may be near 0 and
    # interfere with consistency scoring of the full-run dark matches).
    best_match = _temporal_consistent_anchor(match_results)
    anchor_fid = best_match["frame_id"]
    anchor_idx = best_match["frame_idx"]
    athlete_venue = (best_match["athlete_vx"], best_match["athlete_vy"])

    if verbose:
        print(f"[reg] Anchor: frame {anchor_fid}  NCC={best_match['score']:.3f}  "
              f"scale={best_match['scale']:.4f}  "
              f"venue_topleft=({best_match['venue_x']},{best_match['venue_y']})  "
              f"template=({best_match['template_w']}×{best_match['template_h']})  "
              f"athlete_venue=({athlete_venue[0]:.1f},{athlete_venue[1]:.1f})")

    # Anchor vs GT
    anchor_gt_err = None
    if anchor_fid in gt:
        true_vx, true_vy = gt[anchor_fid]
        anchor_gt_err = float(np.sqrt(
            (athlete_venue[0] - true_vx) ** 2 + (athlete_venue[1] - true_vy) ** 2
        ))
        if verbose:
            print(f"[reg] Anchor GT error: {anchor_gt_err:.1f}px  "
                  f"(true=({true_vx:.0f},{true_vy:.0f}))")

    # Save anchor visualisation
    anchor_bgr = cv2.imread(
        str(frame_dir / frame_files[anchor_idx]), cv2.IMREAD_COLOR
    )
    if anchor_bgr is not None:
        _save_anchor_match_jpg(
            venue_bgr, anchor_bgr, best_match, athlete_venue,
            out_dir / "anchor_match.jpg"
        )
        if verbose:
            print(f"[reg] Saved anchor_match.jpg")

    # Step 4: Build pseudo-GT from all filtered matches + PCHIP calibration
    eval_result: dict = {}
    all_positions: dict[int, tuple[float, float]] = {}

    if cum_flow_by_fid and len(match_results) >= 2:
        # Build calibration set for PCHIP:
        # 1. High-dark frames (min_dark_px_frame × 4): frames deep in rocky terrain
        #    give the most reliable matches.  The ×4 heuristic separates the
        #    "some dark pixels" (transitional) from the "overwhelmingly dark" (rocky).
        # 2. Always include the anchor frame (most temporally consistent match)
        #    regardless of dark_px — it provides coverage in the early/mid flow range.
        dark_cal_threshold = min_dark_px_frame * 4
        high_dark = {m["frame_id"] for m in match_results
                     if m["dark_px"] >= dark_cal_threshold}
        cal_fids = high_dark | {anchor_fid}
        cal_matches_raw = [m for m in match_results if m["frame_id"] in cal_fids]

        # Add the single best sky-edge match as an early-run calibration point.
        # It covers the start-of-run flow range (flow_y ≈ 0 to -400) where
        # dark-feature matches don't exist (snowy venue top).
        # Using only one prevents noisy sky-edge matches from corrupting PCHIP.
        if sky_match_results:
            best_sky_cal = max(sky_match_results, key=lambda m: m["score"])
            cal_matches_raw.append(best_sky_cal)

        # Sort by NCC (descending) and take top 20 before consistency filter
        # to avoid over-fitting on many low-quality calibration points.
        # Always keep the anchor regardless of its NCC rank — it provides
        # essential coverage in the early/mid cumulative-flow range.
        cal_matches_raw.sort(key=lambda m: -m["score"])
        top20 = cal_matches_raw[:20]
        anchor_in_top20 = any(m["frame_id"] == anchor_fid for m in top20)
        if not anchor_in_top20:
            anchor_match_obj = next(
                (m for m in cal_matches_raw if m["frame_id"] == anchor_fid), None
            )
            if anchor_match_obj is not None:
                top20 = top20[:19] + [anchor_match_obj]
        # Always keep the best sky-edge match (covers early flow range).
        if sky_match_results:
            sky_fid = max(sky_match_results, key=lambda m: m["score"])["frame_id"]
            if not any(m["frame_id"] == sky_fid for m in top20):
                sky_obj = next(
                    (m for m in cal_matches_raw if m["frame_id"] == sky_fid), None
                )
                if sky_obj is not None:
                    top20 = top20[:19] + [sky_obj]
        cal_matches_raw = top20
        # Temporal consistency filter on the reduced set
        cal_matches = _filter_consistent_matches(cal_matches_raw, min_consistency_frac=0.5)
        if verbose:
            print(f"[reg] Calibration: {len(match_results)} matches → "
                  f"{len(cal_matches_raw)} high-dark+anchor (top-20 NCC) → "
                  f"{len(cal_matches)} after consistency filter "
                  f"(dark_threshold={dark_cal_threshold:,})")
        pseudo_gt: dict[int, tuple[float, float]] = {
            m["frame_id"]: (m["athlete_vx"], m["athlete_vy"])
            for m in cal_matches
        }
        if verbose:
            print(f"[reg] PCHIP calibration with {len(pseudo_gt)} pseudo-GT points...")

        interp_x, interp_y, fallback, fy_range = _calibrate_pchip(
            cum_flow_by_fid, frame_ids, pseudo_gt, venue_w, venue_h
        )
        if verbose:
            print(f"[reg] PCHIP calibration flow-y range: [{fy_range[0]:.1f}, {fy_range[1]:.1f}]  "
                  f"(linear fallback outside)")

        for fid in frame_ids:
            pos = _predict_venue(
                cum_flow_by_fid, fid, interp_x, interp_y, fallback, venue_w, venue_h,
                fy_range=fy_range,
            )
            if pos is not None:
                all_positions[fid] = pos

        if gt:
            if verbose:
                print(f"\n[reg] Full-run evaluation vs GT ({len(gt)} points):")
            eval_result = _evaluate(all_positions, gt, verbose=verbose)
            if verbose and "mean_err_px" in eval_result:
                print(f"\n[reg] Summary: mean={eval_result['mean_err_px']:.1f}px  "
                      f"median={eval_result['median_err_px']:.1f}px  "
                      f"P90={eval_result['p90_err_px']:.1f}px  "
                      f"within50={eval_result['within_50px']:.0%}")
    elif not cum_flow_by_fid:
        if verbose:
            print("[reg] No cumulative flow — only anchor position computed.")
    else:
        if verbose:
            print("[reg] Too few matches for PCHIP — only anchor position computed.")

    # Video output
    if not no_video and all_positions:
        if verbose:
            print("\n[reg] Rendering venue mapping video...")
        video_out = out_dir / "venue_registration.mp4"
        _write_venue_video(
            frame_dir, frame_files, frame_ids,
            venue_bgr, all_positions, video_out,
            gt=gt if gt else None,
            start_anchor_region=start_anchor_region,
            verbose=verbose,
        )
        if verbose:
            print(f"[reg] Video saved: {video_out}")

    # Save results
    results = {
        "video_stem": stem,
        "venue_size": [venue_w, venue_h],
        "n_tracking_frames": n_frames,
        "n_gt_anchors": len(gt),
        "params": {
            "dark_threshold": dark_threshold,
            "min_gradient": min_gradient,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "n_scales": n_scales,
            "frame_step": frame_step,
            "min_dark_px_frame": min_dark_px_frame,
            "start_anchor_region": list(start_anchor_region) if start_anchor_region else None,
            "search_region": list(search_region) if search_region else None,
        },
        "n_valid_matches": len(match_results),
        "anchor": {
            "frame_id": anchor_fid,
            "ncc_score": best_match["score"],
            "scale": best_match["scale"],
            "venue_topleft": [best_match["venue_x"], best_match["venue_y"]],
            "template_size": [best_match["template_w"], best_match["template_h"]],
            "athlete_venue_pos": list(athlete_venue),
            "gt_error_px": anchor_gt_err,
        },
        "pchip_eval": eval_result if eval_result else None,
        "per_frame_positions": [
            {"frame_id": fid, "venue_x": vx, "venue_y": vy}
            for fid, (vx, vy) in sorted(all_positions.items())
        ],
    }
    results_path = out_dir / "registration_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if verbose:
        print(f"\n[reg] Results saved: {results_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Venue registration via multi-scale dark-patch template matching."
    )
    p.add_argument("video", type=Path, help="Path to the raw video (stem used for outputs)")
    p.add_argument("--venue-image", type=Path, default=Path("venue_image.png"))
    p.add_argument("--tracking-root", type=Path, default=Path("output/tracking"))
    p.add_argument("--velocity-root", type=Path, default=Path("output/velocity"))
    p.add_argument("--frame-root", type=Path, default=Path("output/frames"))
    p.add_argument("--gt-root", type=Path, default=Path("annotations/venue"))
    p.add_argument("--output-root", type=Path, default=Path("output/venue_registration"))
    p.add_argument("--dark-threshold", type=int, default=80)
    p.add_argument("--min-gradient", type=float, default=5.0)
    p.add_argument("--scale-min", type=float, default=0.08)
    p.add_argument("--scale-max", type=float, default=0.15)
    p.add_argument("--n-scales", type=int, default=15)
    p.add_argument("--frame-step", type=int, default=10,
                   help="Match every Nth frame (default=10)")
    p.add_argument("--min-dark-px-frame", type=int, default=50_000,
                   help="Min dark-feature pixels in video frame to attempt matching "
                        "(default=50000; lower if too few matches)")
    p.add_argument(
        "--start-anchor-region", type=str, default=None,
        metavar="X1,Y1,X2,Y2",
        help="Small venue rectangle marking the athlete start area — restricts "
             "early-frame anchor matching only, not the full dark scan. "
             "Auto-loaded from annotations/venue/start_region.json when present. "
             "Example: --start-anchor-region 1000,80,1260,190",
    )
    p.add_argument(
        "--search-region", type=str, default=None,
        metavar="X1,Y1,X2,Y2",
        help="Restrict ALL venue template matching to this corridor. "
             "Use when the full run corridor is known. "
             "Example: --search-region 900,0,1500,1440",
    )
    p.add_argument("--no-video", action="store_true")
    return p


def _parse_region(s: str | None, flag: str) -> tuple[int, int, int, int] | None:
    if not s:
        return None
    parts = [int(v) for v in s.replace(" ", ",").split(",") if v]
    if len(parts) != 4:
        raise ValueError(f"{flag} must be X1,Y1,X2,Y2; got {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    args = _build_parser().parse_args()
    run_venue_registration(
        args.video,
        venue_image_path=args.venue_image,
        tracking_root=args.tracking_root,
        velocity_root=args.velocity_root,
        frame_root=args.frame_root,
        gt_root=args.gt_root,
        output_root=args.output_root,
        dark_threshold=args.dark_threshold,
        min_gradient=args.min_gradient,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        n_scales=args.n_scales,
        frame_step=args.frame_step,
        min_dark_px_frame=args.min_dark_px_frame,
        start_anchor_region=_parse_region(args.start_anchor_region, "--start-anchor-region"),
        search_region=_parse_region(args.search_region, "--search-region"),
        no_video=args.no_video,
    )


if __name__ == "__main__":
    main()
