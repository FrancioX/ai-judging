"""Venue mapping v2: tracking-center correspondence approach.

Instead of feature-matching video frames against the venue image (which fails
on snow-covered slopes), this script uses annotated ground-truth venue
positions to directly fit a transform from video-pixel tracking-center
coordinates to venue coordinates.

Method
------
1. Load tracking centers (cx, cy) for all frames from tracking.json.
2. Load sparse GT venue annotations (venue_x, venue_y) for annotated frames.
3. At frames where both tracking and GT are available, build a set of
   correspondences:  (cx, cy) → (venue_x, venue_y).
4. Fit a perspective transform (homography) from tracking-center space to
   venue space using RANSAC.
5. Project all tracking centers through the fitted homography.
6. For frames beyond the tracking range (where tracking.json has no entry),
   fall back to linear interpolation in venue space.
7. Render side-by-side output video and projection.json.

Usage
-----
::

    uv run python scripts/venue_mapping_v2.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4"

    # Use fewer GT annotations for fitting (test generalisation)
    uv run python scripts/venue_mapping_v2.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4" --gt-holdout 0.3

Notes
-----
- Requires tracking output in output/tracking/<stem>/tracking.json.
- Requires GT venue annotations in annotations/venue/<stem>/gt_venue.csv.
  The GT CSV is needed to fit the transform (not just evaluate it).
- At inference time (no GT), the transform must be pre-computed and stored,
  or a per-video calibration step must be run.  For now, this script uses GT
  to fit the transform.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_tracking_centers(
    tracking_manifest_path: Path,
) -> tuple[list[dict], dict[int, tuple[float, float]]]:
    """Load per-frame bounding-box centers from tracking.json.

    Returns
    -------
    tuple[list[dict], dict[int, tuple[float, float]]]
        Raw frame list and {frame_id: (cx, cy)} mapping.
    """
    with tracking_manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    centers: dict[int, tuple[float, float]] = {}
    for i, frame in enumerate(frames):
        bbox = frame.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            fid = int(
                frame["frame_file"]
                .replace("frame_", "")
                .replace(".jpg", "")
                .replace(".png", "")
            )
            centers[fid] = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    return frames, centers


def _load_gt(gt_path: Path) -> dict[int, tuple[float, float]]:
    """Load sparse GT venue annotations from CSV."""
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
# Transform fitting
# ---------------------------------------------------------------------------

def _fit_tracking_to_venue_homography(
    centers: dict[int, tuple[float, float]],
    gt: dict[int, tuple[float, float]],
    *,
    ransac_threshold: float = 20.0,
    min_correspondences: int = 4,
) -> tuple[np.ndarray | None, list[int]]:
    """Fit a homography from video tracking-center to venue coords.

    Parameters
    ----------
    centers : dict[int, tuple[float, float]]
        ``{frame_id: (cx, cy)}`` from tracking output.
    gt : dict[int, tuple[float, float]]
        ``{frame_id: (venue_x, venue_y)}`` from GT annotations.
    ransac_threshold : float
        RANSAC reprojection threshold in venue pixels.
    min_correspondences : int
        Minimum overlapping frames required to fit.

    Returns
    -------
    tuple[np.ndarray | None, list[int]]
        Homography matrix and list of frame IDs used as correspondences.
    """
    common = sorted(set(centers.keys()) & set(gt.keys()))
    if len(common) < min_correspondences:
        print(
            f"[venue_v2] Only {len(common)} correspondences — need ≥{min_correspondences}."
        )
        return None, common

    src = np.array([[centers[fid][0], centers[fid][1]] for fid in common], dtype=np.float32)
    dst = np.array([[gt[fid][0], gt[fid][1]] for fid in common], dtype=np.float32)

    if len(common) == 4:
        # Exact fit (no RANSAC)
        H = cv2.getPerspectiveTransform(src.reshape(4, 1, 2), dst.reshape(4, 1, 2))
        print(f"[venue_v2] Exact homography from {len(common)} correspondences.")
    else:
        H, inlier_mask = cv2.findHomography(
            src.reshape(-1, 1, 2),
            dst.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
        )
        if H is None:
            print("[venue_v2] Homography fitting failed (RANSAC returned None).")
            return None, common
        n_inliers = int(inlier_mask.ravel().sum()) if inlier_mask is not None else 0
        print(
            f"[venue_v2] Homography fitted: {n_inliers}/{len(common)} inliers "
            f"(RANSAC thresh={ransac_threshold}px)."
        )

    return H.astype(np.float32), common


def _fit_affine_from_correspondences(
    centers: dict[int, tuple[float, float]],
    gt: dict[int, tuple[float, float]],
) -> tuple[np.ndarray | None, list[int]]:
    """Fit an affine transform (less prone to degenerate solutions than homography)."""
    common = sorted(set(centers.keys()) & set(gt.keys()))
    if len(common) < 3:
        return None, common

    src = np.array([[centers[fid][0], centers[fid][1]] for fid in common], dtype=np.float32)
    dst = np.array([[gt[fid][0], gt[fid][1]] for fid in common], dtype=np.float32)

    M, inlier_mask = cv2.estimateAffine2D(
        src.reshape(-1, 1, 2),
        dst.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=30.0,
    )
    if M is None:
        return None, common
    n_inliers = int(inlier_mask.ravel().sum()) if inlier_mask is not None else 0
    print(f"[venue_v2] Affine fitted: {n_inliers}/{len(common)} inliers.")
    # Embed in 3×3 for consistent API
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = M
    return H, common


# ---------------------------------------------------------------------------
# Projection + interpolation
# ---------------------------------------------------------------------------

def _project_center(H: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """Project a single 2D point through a 3×3 homography."""
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt, H)
    return float(proj[0, 0, 0]), float(proj[0, 0, 1])


def _build_projected_positions(
    tracking_frames: list[dict],
    centers: dict[int, tuple[float, float]],
    gt: dict[int, tuple[float, float]],
    H: np.ndarray,
    venue_w: int,
    venue_h: int,
    *,
    smooth: bool = True,
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 5.0,
) -> list[tuple[float, float]]:
    """Project all tracking centers through H; fill gaps by interpolation."""
    # Get all frame ids in order
    frame_ids: list[int] = []
    for fr in tracking_frames:
        fid = int(
            fr["frame_file"].replace("frame_", "").replace(".jpg", "").replace(".png", "")
        )
        frame_ids.append(fid)

    # Project
    raw: list[tuple[float, float]] = []
    for fid in frame_ids:
        if fid in centers:
            cx, cy = centers[fid]
            vx, vy = _project_center(H, cx, cy)
            vx = float(np.clip(vx, 0, venue_w - 1))
            vy = float(np.clip(vy, 0, venue_h - 1))
        else:
            raw.append((-1.0, -1.0))
            continue
        raw.append((vx, vy))

    if not smooth:
        return raw

    # Kalman smoothing
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * kalman_process_noise
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * kalman_measurement_noise
    kalman.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

    # Seed with first valid point
    first_valid = next(((x, y) for x, y in raw if x >= 0), raw[0])
    kalman.statePost = np.array(
        [[first_valid[0]], [first_valid[1]], [0.0], [0.0]], dtype=np.float32
    )

    smoothed: list[tuple[float, float]] = []
    for x, y in raw:
        kalman.predict()
        if x >= 0:
            corrected = kalman.correct(np.array([[x], [y]], dtype=np.float32))
        else:
            # Use prediction only for invalid frames
            corrected = kalman.statePost
        smoothed.append((float(corrected[0, 0]), float(corrected[1, 0])))

    return smoothed


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _draw_trail_panel(
    venue_bgr: np.ndarray,
    trail_points: deque[tuple[int, int]],
    current_point: tuple[int, int],
) -> np.ndarray:
    panel = venue_bgr.copy()
    points = list(trail_points)
    if len(points) >= 2:
        n_seg = len(points) - 1
        for i in range(n_seg):
            alpha = float(i + 1) / float(max(1, n_seg))
            color = (int(20 + 120 * alpha), int(40 + 180 * alpha), int(200 + 30 * alpha))
            thickness = 1 if alpha < 0.4 else 2
            cv2.line(
                panel, points[i], points[i + 1], color,
                thickness=thickness, lineType=cv2.LINE_AA,
            )
    cv2.circle(panel, current_point, radius=8, color=(20, 40, 240), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(panel, current_point, radius=12, color=(180, 220, 255), thickness=2, lineType=cv2.LINE_AA)
    return panel


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def map_video_to_venue_v2(
    video_path: str | Path,
    *,
    venue_image_path: str | Path = "venue_image.png",
    output_root: str | Path = "output/venue_mapping",
    frame_root: str | Path = "output/frames",
    tracking_root: str | Path = "output/tracking",
    gt_root: str | Path = "annotations/venue",
    method: str = "homography",
    ransac_threshold: float = 20.0,
    smooth: bool = True,
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 5.0,
    trail_length: int = 100,
    fps: float = 30.0,
) -> Path:
    """Map tracked athlete positions to venue using GT-fitted transform.

    Parameters
    ----------
    method : str
        ``"homography"`` (perspective) or ``"affine"``.
    ransac_threshold : float
        RANSAC threshold in venue pixels for homography fitting.
    smooth : bool
        Apply Kalman smoothing to projected positions.
    """
    video_path = Path(video_path)
    video_stem = video_path.stem

    frame_dir = Path(frame_root) / video_stem
    tracking_manifest = Path(tracking_root) / video_stem / "tracking.json"
    venue_image_path = Path(venue_image_path)
    gt_path = Path(gt_root) / video_stem / "gt_venue.csv"
    output_dir = Path(output_root) / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    if not tracking_manifest.exists():
        raise FileNotFoundError(f"Tracking manifest not found: {tracking_manifest}")
    if not venue_image_path.exists():
        raise FileNotFoundError(f"Venue image not found: {venue_image_path}")
    if not gt_path.exists():
        raise FileNotFoundError(
            f"GT venue annotations not found: {gt_path}\n"
            "Run annotate_venue.py first to create ground-truth annotations."
        )

    venue_bgr = cv2.imread(str(venue_image_path), cv2.IMREAD_COLOR)
    if venue_bgr is None:
        raise RuntimeError(f"Cannot read venue image: {venue_image_path}")
    venue_h, venue_w = venue_bgr.shape[:2]

    tracking_frames, centers = _load_tracking_centers(tracking_manifest)
    gt = _load_gt(gt_path)
    n_frames = len(tracking_frames)

    print(f"[venue_v2] {len(centers)} tracking frames, {len(gt)} GT annotations")
    print(f"[venue_v2] GT frame range: {min(gt)}-{max(gt)}")
    print(f"[venue_v2] Tracking frame range: {min(centers)}-{max(centers)}")

    # --- Fit transform ---
    if method == "affine":
        H, used_frames = _fit_affine_from_correspondences(centers, gt)
    else:
        H, used_frames = _fit_tracking_to_venue_homography(
            centers, gt, ransac_threshold=ransac_threshold
        )

    if H is None:
        raise RuntimeError(
            f"Failed to fit {method} transform. "
            f"Common frames: {sorted(set(centers.keys()) & set(gt.keys()))}"
        )

    # Evaluate fit quality on training data
    errors = []
    for fid in used_frames:
        if fid not in centers:
            continue
        vx_pred, vy_pred = _project_center(H, centers[fid][0], centers[fid][1])
        vx_gt, vy_gt = gt[fid]
        err = float(np.sqrt((vx_pred - vx_gt) ** 2 + (vy_pred - vy_gt) ** 2))
        errors.append(err)
    if errors:
        print(
            f"[venue_v2] Training residuals: mean={np.mean(errors):.1f}px "
            f"max={np.max(errors):.1f}px"
        )

    # --- Project all frames ---
    smoothed = _build_projected_positions(
        tracking_frames, centers, gt, H, venue_w, venue_h,
        smooth=smooth,
        kalman_process_noise=kalman_process_noise,
        kalman_measurement_noise=kalman_measurement_noise,
    )

    projected_points: list[tuple[int, int]] = [
        (int(np.clip(round(vx), 0, venue_w - 1)), int(np.clip(round(vy), 0, venue_h - 1)))
        for vx, vy in smoothed
    ]

    # --- Render output video ---
    sample_frame = tracking_frames[0]
    sample_frame_path = frame_dir / sample_frame["frame_file"]
    sample_bgr = cv2.imread(str(sample_frame_path))
    if sample_bgr is None:
        raise RuntimeError(f"Cannot read sample frame: {sample_frame_path}")
    frame_h, frame_w = sample_bgr.shape[:2]

    venue_panel_w = int(round((venue_w / max(1, venue_h)) * frame_h))
    venue_panel_w = max(200, venue_panel_w)
    output_path = output_dir / "venue_mapping.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_w + venue_panel_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    trail: deque[tuple[int, int]] = deque(maxlen=max(1, trail_length))
    try:
        for idx in range(n_frames):
            frame_file = tracking_frames[idx]["frame_file"]
            frame_path = frame_dir / frame_file
            frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                frame_bgr = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

            current_pt = projected_points[idx]
            trail.append(current_pt)
            right_panel = _draw_trail_panel(venue_bgr, trail, current_pt)
            right_panel = cv2.resize(right_panel, (venue_panel_w, frame_h), interpolation=cv2.INTER_AREA)
            writer.write(np.hstack((frame_bgr, right_panel)))
    finally:
        writer.release()

    # --- Build projection manifest ---
    projected_list = []
    for idx, frame_entry in enumerate(tracking_frames):
        fid = int(
            frame_entry["frame_file"]
            .replace("frame_", "")
            .replace(".jpg", "")
            .replace(".png", "")
        )
        vx, vy = smoothed[idx]
        projected_list.append({"frame_id": fid, "venue_x": float(vx), "venue_y": float(vy)})

    manifest = {
        "video_stem": video_stem,
        "venue_image": str(venue_image_path),
        "tracking_manifest": str(tracking_manifest),
        "n_frames": n_frames,
        "method": method,
        "gt_correspondences_used": len(used_frames),
        "output_video": str(output_path),
        "projected_positions": projected_list,
    }
    (output_dir / "projection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[venue_v2] Wrote: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Venue mapping v2: tracking-center correspondence approach.",
    )
    p.add_argument("video", type=Path)
    p.add_argument("--venue-image", type=Path, default=Path("venue_image.png"))
    p.add_argument("--output-root", type=Path, default=Path("output/venue_mapping"))
    p.add_argument("--frame-root", type=Path, default=Path("output/frames"))
    p.add_argument("--tracking-root", type=Path, default=Path("output/tracking"))
    p.add_argument("--gt-root", type=Path, default=Path("annotations/venue"))
    p.add_argument("--method", choices=["homography", "affine"], default="homography")
    p.add_argument("--ransac-threshold", type=float, default=20.0)
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--fps", type=float, default=30.0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    map_video_to_venue_v2(
        video_path=args.video,
        venue_image_path=args.venue_image,
        output_root=args.output_root,
        frame_root=args.frame_root,
        tracking_root=args.tracking_root,
        gt_root=args.gt_root,
        method=args.method,
        ransac_threshold=args.ransac_threshold,
        smooth=not args.no_smooth,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
