"""Render velocity overlay video with GT jump intervals highlighted.

Extends the standard velocity overlay with:
- Green band = GT jump interval
- Orange band = predicted jump interval (both shown when they differ)

Usage:
    uv run python scripts/velocity_jump_overlay.py "Ski Men_11_74.83_Emile Peizerat"
    uv run python scripts/velocity_jump_overlay.py "Ski Men_2_89_Andreas Bakke"
"""
from __future__ import annotations

import csv
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_FRAME_ROOT = _ROOT / "output" / "frames"
_VELOCITY_DIR = _ROOT / "output" / "velocity"
_ANNOTATIONS_DIR = _ROOT / "annotations" / "tracking"

_TIMELINE_H = 80
_WHITE = (255, 255, 255)
_GREEN = (0, 200, 0)
_ORANGE = (0, 140, 255)
_RED = (0, 0, 220)
_CYAN = (220, 180, 0)
_YELLOW = (0, 220, 220)
_TRACE = 40


def _load_gt_source_frames(gt_path: Path) -> set[int]:
    """Return set of source frame numbers that fall within any GT jump."""
    s = set()
    with open(gt_path, newline="") as f:
        for row in csv.DictReader(f):
            sf = int(row["start_source_frame"])
            ef = int(row["end_source_frame"])
            if ef > sf:
                s.update(range(sf, ef + 1))
    return s


def main(stem: str) -> None:
    velocity_dir = _VELOCITY_DIR / stem
    frame_dir = _FRAME_ROOT / stem
    gt_path = _ANNOTATIONS_DIR / stem / "gt_jumps.csv"
    out_path = velocity_dir / "jump_overlay.mp4"

    vel_json = velocity_dir / "velocity.json"
    with open(vel_json) as f:
        vel_data = json.load(f)

    frames_list = vel_data["frames"]
    # Index by source frame number (number in filename)
    vel_by_sfid: dict[int, dict] = {
        int(Path(fr["frame_file"]).stem.split("_")[-1]): fr
        for fr in frames_list
    }

    # Predicted jump set (source frame numbers)
    pred_jump_set: set[int] = set()
    for jmp in vel_data.get("jumps", []):
        s = int(Path(frames_list[jmp["start_idx"]]["frame_file"]).stem.split("_")[-1])
        e = int(Path(frames_list[jmp["end_idx"]]["frame_file"]).stem.split("_")[-1])
        pred_jump_set.update(range(s, e + 1))

    # GT jump set
    gt_jump_set: set[int] = set()
    if gt_path.exists():
        gt_jump_set = _load_gt_source_frames(gt_path)

    max_speed = vel_data["summary"].get("max_speed", 20.0) or 20.0

    frame_files = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(frame_dir.glob("frame_*.png"))

    fps = 25.0
    meta = frame_dir / "frames_meta.json"
    if meta.exists():
        with open(meta) as f:
            fps = json.load(f).get("fps", 25.0)

    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]
    out_h = h + _TIMELINE_H

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, out_h))

    all_speeds = [
        float(vel_by_sfid[int(p.stem.split("_")[-1])]["speed_smoothed"])
        if int(p.stem.split("_")[-1]) in vel_by_sfid else 0.0
        for p in frame_files
    ]
    center_trace: deque[tuple[int, int]] = deque(maxlen=_TRACE)

    print(f"Rendering {stem} → {out_path.name}")
    for frame_idx, frame_path in enumerate(frame_files):
        sfid = int(frame_path.stem.split("_")[-1])
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        canvas = np.zeros((out_h, w, 3), dtype=np.uint8)
        canvas[:h] = img

        in_gt = sfid in gt_jump_set
        in_pred = sfid in pred_jump_set

        # Tint: green=GT, orange=pred, yellow=both
        if in_gt and in_pred:
            overlay = canvas[:h].copy()
            cv2.rectangle(overlay, (0, 0), (w, h), _YELLOW, -1)
            cv2.addWeighted(overlay, 0.15, canvas[:h], 0.85, 0, canvas[:h])
        elif in_gt:
            overlay = canvas[:h].copy()
            cv2.rectangle(overlay, (0, 0), (w, h), _GREEN, -1)
            cv2.addWeighted(overlay, 0.15, canvas[:h], 0.85, 0, canvas[:h])
        elif in_pred:
            overlay = canvas[:h].copy()
            cv2.rectangle(overlay, (0, 0), (w, h), _ORANGE, -1)
            cv2.addWeighted(overlay, 0.10, canvas[:h], 0.90, 0, canvas[:h])

        vf = vel_by_sfid.get(sfid)
        if vf:
            ax = int(vf.get("center_x", w / 2))
            ay = int(vf.get("center_y", h / 2))
            bb = vf.get("bbox")
            if bb:
                cv2.rectangle(canvas, (bb[0], bb[1]), (bb[2], bb[3]), _CYAN, 2, cv2.LINE_AA)
            dvx = float(vf["compensated_vel_dx"])
            dvy = float(vf["compensated_vel_dy"])
            ex = int(ax + dvx * 5)
            ey = int(ay + dvy * 5)
            cv2.arrowedLine(canvas, (ax, ay), (ex, ey), _YELLOW, 2, cv2.LINE_AA, tipLength=0.3)
            center_trace.append((ax, ay))

        # Trace
        pts = list(center_trace)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            color = (int(220 * alpha), int(180 * alpha), 0)
            cv2.line(canvas, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

        # HUD
        speed = all_speeds[frame_idx]
        label = "GT+PRED" if (in_gt and in_pred) else ("GT" if in_gt else ("PRED" if in_pred else ""))
        color = _YELLOW if (in_gt and in_pred) else (_GREEN if in_gt else (_ORANGE if in_pred else _WHITE))
        cv2.putText(canvas, f"fr={sfid}  spd={speed:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _WHITE, 2, cv2.LINE_AA)
        if label:
            cv2.putText(canvas, f"JUMP: {label}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Timeline strip
        tl = canvas[h:, :]
        tl[:] = 20
        n = len(all_speeds)
        for i, sp in enumerate(all_speeds):
            x = int(i / n * w)
            bar_h = int(sp / (max_speed + 1e-6) * (_TIMELINE_H - 4))
            sfid_i = int(frame_files[i].stem.split("_")[-1])
            if sfid_i in gt_jump_set:
                c = _GREEN
            elif sfid_i in pred_jump_set:
                c = _ORANGE
            else:
                ratio = min(1.0, sp / (max_speed + 1e-6))
                g = 220 if ratio < 0.5 else int((1 - (ratio - 0.5) * 2) * 220)
                r = int(ratio * 2 * 220) if ratio < 0.5 else 220
                c = (0, g, r)
            cv2.line(tl, (x, _TIMELINE_H - 2), (x, _TIMELINE_H - 2 - bar_h), c, 1)

        # Playhead
        px = int(frame_idx / n * w)
        cv2.line(canvas[h:], (px, 0), (px, _TIMELINE_H), _WHITE, 1)

        writer.write(canvas)
        if frame_idx % 200 == 0:
            print(f"  {frame_idx}/{len(frame_files)}")

    writer.release()
    print(f"Done → {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/velocity_jump_overlay.py <video_stem>")
        sys.exit(1)
    main(sys.argv[1])
