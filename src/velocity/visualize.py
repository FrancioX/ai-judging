"""Velocity overlay visualization for the velocity pipeline stage.

Renders ``velocity_overlay.mp4`` on top of the original frames with:
- Speed gauge (color-coded bar)
- Velocity arrow on the athlete
- Jump interval highlighting
- Speed timeline strip at the bottom
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRACE_LENGTH = 40

# BGR colors
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GREEN = (0, 220, 0)
_YELLOW = (0, 220, 220)
_RED = (0, 0, 220)
_ORANGE = (0, 140, 255)
_CYAN = (220, 180, 0)
_BLUE = (220, 80, 0)
_BG = (0, 0, 0)

_TIMELINE_H = 60    # pixels reserved for the speed timeline strip
_GAUGE_W = 180      # width of the speed gauge bar
_GAUGE_H = 18       # height of the speed gauge bar


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_velocity_overlay(
    velocity_dir: str | Path,
    frame_root: str | Path,
    *,
    output_path: Path | None = None,
    fps: float | None = None,
    trace_length: int = _TRACE_LENGTH,
) -> Path:
    """Render velocity overlay video.

    Parameters
    ----------
    velocity_dir : str | Path
        Directory containing ``velocity.json``.
    frame_root : str | Path
        Root frames directory; the video-stem subdirectory is inferred from
        *velocity_dir*.
    output_path : Path | None
        Output MP4 path. Defaults to ``<velocity_dir>/velocity_overlay.mp4``.
    fps : float | None
        Output FPS. Auto-detected from ``frames_meta.json`` if None.
    trace_length : int
        Trail length for the velocity trace.

    Returns
    -------
    Path
        Path to the rendered MP4.
    """
    velocity_dir = Path(velocity_dir)
    video_stem = velocity_dir.name
    frame_dir = Path(frame_root) / video_stem

    vel_json = velocity_dir / "velocity.json"
    if not vel_json.exists():
        raise FileNotFoundError(f"velocity.json not found: {vel_json}")

    with open(vel_json) as f:
        vel_data = json.load(f)

    if vel_data.get("stub"):
        raise RuntimeError(f"velocity.json is a stub (no real data): {vel_json}")

    frames_list = vel_data.get("frames", [])
    if not frames_list:
        raise RuntimeError("velocity.json has no frames")

    # Index by the number in the frame filename (e.g. frame_000150.jpg → 150)
    # so it matches what _fid_of() extracts from the frame directory listing.
    vel_by_fid: dict[int, dict] = {
        int(Path(fr["frame_file"]).stem.split("_")[-1]): fr
        for fr in frames_list
    }
    jump_set: set[int] = set()
    for jmp in vel_data.get("jumps", []):
        # Convert sequential frame indices back to filename numbers via the list
        s = int(Path(frames_list[jmp["start_idx"]]["frame_file"]).stem.split("_")[-1])
        e = int(Path(frames_list[jmp["end_idx"]]["frame_file"]).stem.split("_")[-1])
        jump_set.update(range(s, e + 1))

    max_speed = vel_data["summary"].get("max_speed", 20.0) or 20.0

    # Discover ordered frames
    frame_files = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(frame_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    fps = fps or _detect_fps(frame_dir) or 25.0

    if output_path is None:
        output_path = velocity_dir / "velocity_overlay.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]
    out_h = h + _TIMELINE_H

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, out_h))

    # Pre-build full speed array (indexed by position in frame_files)
    all_speeds = [
        float(vel_by_fid[_fid_of(p)]["speed_smoothed"])
        if _fid_of(p) in vel_by_fid else 0.0
        for p in frame_files
    ]
    # Athlete center trace
    center_trace: deque[tuple[int, int]] = deque(maxlen=trace_length)

    def _fid(p: Path) -> int:
        return int(p.stem.split("_")[-1])

    print(f"\n[velocity] Rendering overlay: {video_stem}")

    for frame_idx, frame_path in enumerate(frame_files):
        fid = _fid(frame_path)
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        vf = vel_by_fid.get(fid)
        speed = all_speeds[frame_idx]

        # ── Canvas: frame + timeline strip ──────────────────────────────
        canvas = np.zeros((out_h, w, 3), dtype=np.uint8)
        canvas[:h] = img

        in_jump = fid in jump_set
        if in_jump:
            overlay = canvas[:h].copy()
            cv2.rectangle(overlay, (0, 0), (w, h), _ORANGE, -1)
            cv2.addWeighted(overlay, 0.10, canvas[:h], 0.90, 0, canvas[:h])

        if vf:
            ax = int(vf.get("center_x", w / 2))
            ay = int(vf.get("center_y", h / 2))

            # Draw tracking bbox
            bb = vf.get("bbox")
            if bb:
                cv2.rectangle(canvas, (bb[0], bb[1]), (bb[2], bb[3]), _CYAN, 2, cv2.LINE_AA)

            # Draw velocity arrow from bbox center
            dvx = float(vf["compensated_vel_dx"])
            dvy = float(vf["compensated_vel_dy"])
            scale = 5.0
            ex = int(ax + dvx * scale)
            ey = int(ay + dvy * scale)
            cv2.arrowedLine(canvas, (ax, ay), (ex, ey), _YELLOW, 2, cv2.LINE_AA, tipLength=0.3)
            center_trace.append((ax, ay))

        _draw_speed_gauge(canvas, speed, max_speed, w, in_jump)
        _draw_timeline(canvas, all_speeds, frame_idx, h, w, max_speed)
        _draw_hud(canvas, fid, speed, in_jump, w, vel_data)

        writer.write(canvas)

    writer.release()
    print(f"  Saved → {output_path}  ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _speed_color(speed: float, max_speed: float) -> tuple[int, int, int]:
    """Return BGR color interpolated green → yellow → red by speed ratio."""
    ratio = min(1.0, speed / (max_speed + 1e-6))
    if ratio < 0.5:
        g = 220
        r = int(ratio * 2 * 220)
        return (0, g, r)
    else:
        r = 220
        g = int((1.0 - (ratio - 0.5) * 2) * 220)
        return (0, g, r)


def _draw_speed_gauge(
    canvas: np.ndarray,
    speed: float,
    max_speed: float,
    img_w: int,
    in_jump: bool,
) -> None:
    """Draw a speed gauge bar in the top-right corner."""
    x0 = img_w - _GAUGE_W - 10
    y0 = 10
    x1 = img_w - 10
    y1 = y0 + _GAUGE_H

    # Background
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (50, 50, 50), -1)

    # Fill
    ratio = min(1.0, speed / (max_speed + 1e-6))
    fill_x = x0 + int(ratio * _GAUGE_W)
    color = _speed_color(speed, max_speed)
    if fill_x > x0:
        cv2.rectangle(canvas, (x0, y0), (fill_x, y1), color, -1)

    # Border
    cv2.rectangle(canvas, (x0, y0), (x1, y1), _WHITE, 1)

    label = f"{speed:.1f} px/f"
    if in_jump:
        label += "  JUMP"
    cv2.putText(
        canvas, label,
        (x0, y1 + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color if not in_jump else _ORANGE,
        1, cv2.LINE_AA,
    )


def _draw_timeline(
    canvas: np.ndarray,
    all_speeds: list[float],
    current_idx: int,
    frame_h: int,
    img_w: int,
    max_speed: float,
) -> None:
    """Draw a fixed speed timeline strip showing the full run with a cursor.

    The full speed curve is drawn once across the entire strip width.
    A white vertical line marks the current frame position.
    """
    strip_y = frame_h
    strip_h = _TIMELINE_H
    pad_top = 6
    pad_bot = 5
    usable_h = strip_h - pad_top - pad_bot

    cv2.rectangle(canvas, (0, strip_y), (img_w, strip_y + strip_h), (20, 20, 20), -1)

    n = len(all_speeds)
    if n < 2:
        return

    def _y(s: float) -> int:
        return strip_y + strip_h - pad_bot - int(s / (max_speed + 1e-6) * usable_h)

    def _x(i: int) -> int:
        return int(i / (n - 1) * (img_w - 1))

    # Draw the full speed polyline
    for i in range(1, n):
        x_prev, x_curr = _x(i - 1), _x(i)
        y_prev = _y(all_speeds[i - 1])
        y_curr = _y(all_speeds[i])
        color = _speed_color(all_speeds[i], max_speed)
        cv2.line(canvas, (x_prev, y_prev), (x_curr, y_curr), color, 1, cv2.LINE_AA)

    # Cursor: vertical white line at current position
    cx = _x(current_idx)
    cv2.line(canvas, (cx, strip_y + 2), (cx, strip_y + strip_h - 2), _WHITE, 1, cv2.LINE_AA)

    # Label
    cv2.putText(
        canvas, "speed",
        (4, strip_y + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA,
    )


def _draw_hud(
    canvas: np.ndarray,
    frame_id: int,
    speed: float,
    in_jump: bool,
    img_w: int,
    vel_data: dict,
) -> None:
    """Draw minimal HUD in top-left."""
    cv2.rectangle(canvas, (0, 0), (200, 52), _BG, -1)
    cv2.putText(
        canvas, f"Frame {frame_id}",
        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA,
    )
    n_jumps = vel_data.get("summary", {}).get("n_jumps", 0)
    cv2.putText(
        canvas, f"spd:{speed:.1f}  jumps:{n_jumps}",
        (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CYAN, 1, cv2.LINE_AA,
    )
    if in_jump:
        cv2.putText(
            canvas, "IN JUMP",
            (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _ORANGE, 2, cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fid_of(p: Path) -> int:
    return int(p.stem.split("_")[-1])


def _detect_fps(frame_dir: Path) -> float | None:
    meta_path = frame_dir / "frames_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("effective_fps") or meta.get("source_fps")
    return None
