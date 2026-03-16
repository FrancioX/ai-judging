"""Athlete velocity extraction from monocular video.

Estimates ground-relative velocity by subtracting camera pan (estimated via
background optical flow on dark rock/tree features) from the tracked pixel
velocity:

    v_ground = v_athlete_pixel - v_camera_pan
             = v_athlete_pixel + v_background_flow

Outputs per-frame velocity, speed, acceleration, jump intervals, and a
trajectory-smoothness metric.

Public API
----------
    extract_velocity(tracking_dir, output_dir, *, frame_root, **config) -> Path
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from src.utils.optical_flow import compute_per_frame_background_flow
    _OF_AVAILABLE = True
except ImportError:
    _OF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_velocity(
    tracking_dir: str | Path,
    output_dir: str | Path,
    *,
    frame_root: str | Path,
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
    min_dark_pixels: int = 50,
    smooth_window: int = 5,
    zoom_correct: bool = False,
    normalize_by_height: bool = False,
    jump_min_airtime_frames: int = 5,
    jump_vy_threshold: float = 3.0,
) -> Path:
    """Extract athlete ground-relative velocity from tracking output.

    Parameters
    ----------
    tracking_dir : str | Path
        Path to ``output/tracking/<stem>/`` containing ``tracking.json``.
    output_dir : str | Path
        Destination for ``velocity.json``.
    frame_root : str | Path
        Root frames directory, e.g. ``output/frames``. The video-stem
        subdirectory is inferred from *tracking_dir*.
    dark_threshold : int
        Brightness ceiling for "dark feature" pixels (rocks/trees).
    min_gradient : float
        Minimum Sobel gradient magnitude to qualify as textured.
    min_dark_pixels : int
        Minimum dark-feature pixels required; below this falls back to
        all-background median flow.
    smooth_window : int
        Rolling-average half-window for speed smoothing (0 = off).
    zoom_correct : bool
        Normalise background flow by bbox-area proxy for camera zoom.
    normalize_by_height : bool
        Divide compensated speed by bbox height for scale invariance.
    jump_min_airtime_frames : int
        Minimum duration (frames) for a detected jump interval.
    jump_vy_threshold : float
        Vertical velocity threshold (px/frame) that triggers jump detection.

    Returns
    -------
    Path
        Path to written ``velocity.json``.
    """
    tracking_dir = Path(tracking_dir)
    output_dir = Path(output_dir)
    video_stem = tracking_dir.name
    frame_dir = Path(frame_root) / video_stem

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "velocity.json"

    if not _CV2_AVAILABLE or not _OF_AVAILABLE:
        return _write_stub(out_path, video_stem, reason="opencv or optical-flow module unavailable")

    # ── Load tracking manifest ──────────────────────────────────────────
    manifest_path = tracking_dir / "tracking.json"
    if not manifest_path.exists():
        return _write_stub(out_path, video_stem, reason=f"tracking.json not found: {manifest_path}")

    with open(manifest_path) as f:
        tracking_data = json.load(f)

    frames_data = tracking_data.get("frames", [])
    if not frames_data:
        return _write_stub(out_path, video_stem, reason="tracking.json has no frames")

    frame_files = [fr["frame_file"] for fr in frames_data]
    frame_ids = [fr.get("frame_id", i) for i, fr in enumerate(frames_data)]
    bboxes = [fr.get("bbox_padded") or fr.get("bbox") or [0, 0, 64, 64] for fr in frames_data]
    tight_bboxes = [fr.get("bbox") or fr.get("bbox_padded") or [0, 0, 64, 64] for fr in frames_data]

    # ── Bbox centers (athlete pixel positions) ──────────────────────────
    centers = np.array(
        [
            [
                (bb[0] + bb[2]) / 2.0,
                (bb[1] + bb[3]) / 2.0,
            ]
            for bb in bboxes
        ],
        dtype=float,
    )

    # Bbox heights for optional normalisation
    bbox_heights = np.array(
        [max(1.0, bb[3] - bb[1]) for bb in bboxes],
        dtype=float,
    )

    # ── Per-frame athlete pixel velocity ────────────────────────────────
    n = len(frame_files)
    athlete_vel = np.zeros((n, 2), dtype=float)
    athlete_vel[1:] = centers[1:] - centers[:-1]

    # ── Background flow (camera pan) ────────────────────────────────────
    print(f"  [velocity] Computing background flow for {video_stem}...")
    cam_flow, n_dark = compute_per_frame_background_flow(
        frame_dir,
        frame_files,
        bboxes,
        dark_threshold=dark_threshold,
        min_gradient=min_gradient,
        min_dark_pixels=min_dark_pixels,
        zoom_correct=zoom_correct,
        verbose=True,
    )

    # ── Ground-relative velocity ─────────────────────────────────────────
    # v_ground = v_athlete_pixel - v_background_flow
    # Background flow is the apparent scene motion in the image (opposite to
    # camera pan direction). Subtracting it removes the camera contribution:
    # if camera pans right (bg_flow_dx < 0), athlete's ground speed is higher
    # than its pixel speed → v_ground_dx = v_pixel_dx - bg_flow_dx (adds the
    # camera pan contribution back in).
    compensated_vel = athlete_vel - cam_flow

    # Optional bbox-height normalisation
    if normalize_by_height:
        compensated_vel = compensated_vel / bbox_heights[:, None]

    # Speed (scalar)
    raw_speed = np.linalg.norm(compensated_vel, axis=1)

    # Smooth speed
    if smooth_window > 0:
        kernel = np.ones(smooth_window * 2 + 1) / (smooth_window * 2 + 1)
        smoothed_speed = np.convolve(raw_speed, kernel, mode="same")
    else:
        smoothed_speed = raw_speed.copy()

    # ── Acceleration & jerk ─────────────────────────────────────────────
    accel = np.zeros(n, dtype=float)
    accel[1:] = smoothed_speed[1:] - smoothed_speed[:-1]

    jerk = np.zeros(n, dtype=float)
    jerk[1:] = accel[1:] - accel[:-1]

    smoothness = float(1.0 / (np.mean(np.abs(jerk)) + 1e-6))

    # Direction curvature (angle change per frame)
    angles = np.arctan2(compensated_vel[:, 1], compensated_vel[:, 0] + 1e-9)
    angle_diff = np.zeros(n, dtype=float)
    angle_diff[1:] = np.abs(
        np.arctan2(
            np.sin(angles[1:] - angles[:-1]),
            np.cos(angles[1:] - angles[:-1]),
        )
    )
    mean_curvature = float(np.mean(angle_diff))

    # ── Jump detection ───────────────────────────────────────────────────
    jumps = _detect_jumps(
        compensated_vel,
        frame_ids,
        jump_vy_threshold=jump_vy_threshold,
        min_airtime_frames=jump_min_airtime_frames,
    )

    # ── Assemble per-frame records ───────────────────────────────────────
    jump_frame_set = set()
    for jmp in jumps:
        jump_frame_set.update(range(jmp["start_idx"], jmp["end_idx"] + 1))

    per_frame = []
    for i in range(n):
        bb = tight_bboxes[i]
        per_frame.append(
            {
                "frame_id": int(frame_ids[i]),
                "frame_file": frame_files[i],
                "center_x": round(float(centers[i, 0]), 2),
                "center_y": round(float(centers[i, 1]), 2),
                "bbox": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
                "camera_flow_dx": round(float(cam_flow[i, 0]), 4),
                "camera_flow_dy": round(float(cam_flow[i, 1]), 4),
                "athlete_vel_dx": round(float(athlete_vel[i, 0]), 4),
                "athlete_vel_dy": round(float(athlete_vel[i, 1]), 4),
                "compensated_vel_dx": round(float(compensated_vel[i, 0]), 4),
                "compensated_vel_dy": round(float(compensated_vel[i, 1]), 4),
                "speed_raw": round(float(raw_speed[i]), 4),
                "speed_smoothed": round(float(smoothed_speed[i]), 4),
                "acceleration": round(float(accel[i]), 4),
                "in_jump": i in jump_frame_set,
            }
        )

    # ── Output JSON ──────────────────────────────────────────────────────
    result = {
        "video_stem": video_stem,
        "n_frames": n,
        "n_dark_frames": int(n_dark),
        "config": {
            "dark_threshold": dark_threshold,
            "min_gradient": min_gradient,
            "min_dark_pixels": min_dark_pixels,
            "smooth_window": smooth_window,
            "zoom_correct": zoom_correct,
            "normalize_by_height": normalize_by_height,
            "jump_min_airtime_frames": jump_min_airtime_frames,
            "jump_vy_threshold": jump_vy_threshold,
        },
        "summary": {
            "mean_speed": round(float(np.mean(raw_speed)), 4),
            "max_speed": round(float(np.max(raw_speed)), 4),
            "smoothness": round(smoothness, 6),
            "mean_curvature_rad": round(mean_curvature, 6),
            "n_jumps": len(jumps),
        },
        "jumps": jumps,
        "frames": per_frame,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"  [velocity] {video_stem}: {n} frames, "
        f"mean speed={result['summary']['mean_speed']:.1f} px/frame, "
        f"{len(jumps)} jump(s) detected → {out_path}"
    )
    return out_path


# ---------------------------------------------------------------------------
# Jump detection
# ---------------------------------------------------------------------------

def _detect_jumps(
    compensated_vel: np.ndarray,
    frame_ids: list[int],
    *,
    jump_vy_threshold: float = 3.0,
    min_airtime_frames: int = 5,
) -> list[dict]:
    """Detect jump intervals from compensated vertical velocity.

    A jump is a sustained upward phase (negative vy = upward in image coords)
    followed by a downward phase, lasting at least *min_airtime_frames*.

    Returns a list of dicts with keys:
        start_idx, end_idx, peak_idx,
        start_frame_id, end_frame_id, peak_frame_id,
        airtime_frames, peak_speed_px_per_frame
    """
    vy = compensated_vel[:, 1]  # positive = downward in image
    n = len(vy)

    # "In air" = vy < -jump_vy_threshold (moving upward faster than threshold)
    in_launch = vy < -jump_vy_threshold

    jumps = []
    i = 0
    while i < n:
        if in_launch[i]:
            start = i
            # Extend through the full airborne interval (launch + peak + landing)
            # Allow sign reversal once (going up then coming down)
            j = i + 1
            while j < n and (vy[j] < jump_vy_threshold):
                j += 1
            end = j - 1

            airtime = end - start + 1
            if airtime >= min_airtime_frames:
                peak = int(start + np.argmin(vy[start : end + 1]))
                jumps.append(
                    {
                        "start_idx": start,
                        "end_idx": end,
                        "peak_idx": peak,
                        "start_frame_id": int(frame_ids[start]),
                        "end_frame_id": int(frame_ids[end]),
                        "peak_frame_id": int(frame_ids[peak]),
                        "airtime_frames": airtime,
                        "peak_speed_px_per_frame": round(
                            float(np.linalg.norm(compensated_vel[peak])), 4
                        ),
                    }
                )
            i = j
        else:
            i += 1

    return jumps


# ---------------------------------------------------------------------------
# Stub writer
# ---------------------------------------------------------------------------

def _write_stub(out_path: Path, video_stem: str, *, reason: str) -> Path:
    """Write an empty velocity stub when dependencies are unavailable."""
    print(f"  [velocity] Stub output for {video_stem}: {reason}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stub = {
        "video_stem": video_stem,
        "stub": True,
        "reason": reason,
        "n_frames": 0,
        "summary": {},
        "jumps": [],
        "frames": [],
    }
    with open(out_path, "w") as f:
        json.dump(stub, f, indent=2)
    return out_path
