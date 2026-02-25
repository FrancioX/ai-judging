"""Visualise pure Lucas-Kanade optical-flow tracking precision.

Loads the Lach Powell tracking manifest, picks the bbox centre at the midpoint
of the longest consecutive detected run, and tracks that single point forward
and backward using only optical flow on the extracted frames.

Renders a comparison video:
  - Yellow trace: optical-flow tracked point
  - Green trace: detection-based bbox centres (ground truth reference)
  - HUD shows per-frame pixel drift

Usage:
    uv run python scripts/optical_flow_precision.py

Experiment Results (2026-02-25)
================================
Video: Lach Powell — VERBIER FREERIDE WEEK QUALIFIER 4, Heat 3
  1450 tracking frames (frame_000150–frame_001599), ~30 fps.

Setup:
  - Seed point: idx 319 (frame_000469.jpg), centre of tracking bbox (646, 285).
    Chosen as midpoint of the longest consecutive detected run (idx 99–539,
    441 frames with no interpolation).
  - Lucas-Kanade params: winSize=21×21, maxLevel=3, 30 iters, eps=0.01.
  - OF tracked forward 477 frames, backward 320 frames → 796 total.

Key finding — OF outperforms tracker in the identity-switch zone:
  At idx 448–451 (frames 598–601, ~15s mark) the detection-based tracker
  jumps 451px to a *different skier* entering frame-right. The tracking bbox
  snaps from (966, 548) to (1537, 529) in a single frame. The tracker then
  follows this wrong person from idx 450 all the way through idx ~545.

  Meanwhile, optical flow stays locked on Lach Powell through this entire zone
  because it tracks pixel appearance, not detections. At idx 450 OF reports
  (968, 556) — only ~2px from the previous frame — while the tracker is 569px
  away on the wrong skier.

Phase-by-phase drift analysis (OF vs tracking bbox):

  Window       Drift(mean)  Drift(med)  Notes
  ─────────────────────────────────────────────────────────────────
  idx   0– 49    121 px      123 px     Backward from seed — drift accumulated
  idx  50– 99     33 px       33 px     Backward, still reasonable
  idx 100–149     21 px       22 px     Close to seed, low drift
  idx 150–199     12 px       12 px     ↓
  idx 200–249      7 px        6 px     Near seed — excellent agreement
  idx 250–349    3–6 px      4–6 px     **SEED ZONE**: near-perfect match
  idx 350–399     19 px       21 px     Slight natural divergence
  idx 400–449     16 px       13 px     Tracker starts drifting to other skier
  idx 450–499   **668 px**    682 px    🔴 TRACKER ON WRONG SKIER — OF correct
  idx 500–549   **459 px**    519 px    🔴 Tracker still wrong, 7 interpolated
  idx 550–599     38 px       39 px     Tracker recovers, OF has some drift
  idx 600–699   29–32 px    32–35 px    Moderate divergence
  idx 700–795   90–506 px   40–528 px   OF accumulates drift; heavy interpolation

Smoothness comparison:
  OF:       mean displacement = 4.56 px/frame, std = 2.71
  Tracker:  mean displacement = 6.01 px/frame, std = 17.92
  → OF produces **6.6× smoother** trajectories (lower std) because it has no
    detection noise or identity switches.

Conclusions:
  1. Pure OF is dramatically smoother frame-to-frame (std 2.7 vs 17.9).
  2. OF correctly maintains identity through the idx 448–545 zone where the
     YOLO-based tracker switches to the wrong person (~100 frames affected).
  3. OF accumulates drift over long distances: excellent within ±150 frames
     of seed, but unreliable beyond ~300 frames without re-anchoring.
  4. **Hybrid approach recommended**: use OF for short-range interpolation
     and identity continuity validation, re-anchor to detections periodically.
     The tracker's identity switch at idx 450 could have been prevented by
     checking that the OF-predicted position is far from the new detection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VIDEO_STEM = (
    "VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86"
)
OUTPUT_BASE = Path("output")
FRAMES_DIR = OUTPUT_BASE / "frames" / VIDEO_STEM
TRACKING_JSON = OUTPUT_BASE / "tracking" / VIDEO_STEM / "tracking.json"

# Lucas-Kanade parameters
LK_PARAMS: dict = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Drawing
TRACE_COLOR = (0, 255, 255)   # yellow (BGR)
GT_COLOR = (0, 255, 0)        # green
POINT_COLOR = (0, 0, 255)     # red
SEED_COLOR = (255, 0, 255)    # magenta
LOST_COLOR = (0, 0, 180)      # dark red
TRACE_THICKNESS = 2
POINT_RADIUS = 5
MAX_TRACE_TAIL = 400          # visible tail length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_centre(bbox: list[int]) -> tuple[float, float]:
    """Return (cx, cy) from [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _load_gray(frame_path: Path) -> np.ndarray:
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {frame_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _load_bgr(frame_path: Path) -> np.ndarray:
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {frame_path}")
    return img


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def find_longest_detected_run(
    frames: list[dict],
) -> tuple[int, int]:
    """Return (start_index, length) of the longest consecutive detected run."""
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for i, fr in enumerate(frames):
        if fr["detected"] and not fr["interpolated"]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
            cur_len = 0
    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start
    return best_start, best_len


def track_optical_flow(
    frames_list: list[dict],
    start_idx: int,
    seed_pt: tuple[float, float],
    direction: int = 1,
) -> list[tuple[int, float, float, bool]]:
    """Track *seed_pt* through sequential frames using LK optical flow.

    Parameters
    ----------
    frames_list : list of tracking-JSON frame dicts (must have ``frame_file``).
    start_idx   : index into *frames_list* where the seed lives.
    seed_pt     : (x, y) in pixel coordinates.
    direction   : +1 forward, -1 backward.

    Returns
    -------
    List of (frame_list_index, x, y, ok) tuples.
    """
    results: list[tuple[int, float, float, bool]] = []
    pt = np.array([[list(seed_pt)]], dtype=np.float32)  # (1,1,2)

    prev_path = FRAMES_DIR / frames_list[start_idx]["frame_file"]
    prev_gray = _load_gray(prev_path)
    results.append((start_idx, float(pt[0, 0, 0]), float(pt[0, 0, 1]), True))

    idx = start_idx
    while True:
        idx += direction
        if idx < 0 or idx >= len(frames_list):
            break

        curr_path = FRAMES_DIR / frames_list[idx]["frame_file"]
        if not curr_path.exists():
            break

        curr_gray = _load_gray(curr_path)
        new_pt, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pt, None, **LK_PARAMS,
        )

        ok = bool(status[0, 0])
        if ok:
            pt = new_pt
            results.append((idx, float(pt[0, 0, 0]), float(pt[0, 0, 1]), True))
        else:
            results.append((idx, float(pt[0, 0, 0]), float(pt[0, 0, 1]), False))
            break  # lost

        prev_gray = curr_gray

    return results


def render_comparison_video(
    frames_list: list[dict],
    of_trace: list[tuple[int, float, float, bool]],
    gt_map: dict[int, tuple[float, float]],
    output_path: Path,
    fps: float,
    seed_idx: int,
    scale: float = 0.5,
) -> None:
    """Write an MP4 with OF trace (yellow) and bbox centres (green)."""
    of_trace = sorted(of_trace, key=lambda t: t[0])
    if not of_trace:
        raise RuntimeError("Empty optical-flow trace")

    min_idx = of_trace[0][0]
    max_idx = of_trace[-1][0]

    # Determine frame size from first frame
    first_bgr = _load_bgr(FRAMES_DIR / frames_list[min_idx]["frame_file"])
    orig_h, orig_w = first_bgr.shape[:2]
    h, w = int(orig_h * scale), int(orig_w * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # Pre-scale the OF and GT coordinates
    of_map: dict[int, tuple[float, float, bool]] = {
        t[0]: (t[1] * scale, t[2] * scale, t[3]) for t in of_trace
    }
    gt_map_scaled: dict[int, tuple[float, float]] = {
        k: (v[0] * scale, v[1] * scale) for k, v in gt_map.items()
    }

    of_acc: list[tuple[int, int, bool]] = []
    gt_acc: list[tuple[int, int]] = []
    total = max_idx - min_idx + 1

    import time
    t0 = time.time()

    for count, idx in enumerate(range(min_idx, max_idx + 1)):
        if count % 50 == 0:
            elapsed = time.time() - t0
            pct = count / total * 100
            eta = (elapsed / max(count, 1)) * (total - count)
            print(f"  Rendering {count}/{total} ({pct:.0f}%) ETA {eta:.0f}s …",
                  end="\r", flush=True)

        frame_file = FRAMES_DIR / frames_list[idx]["frame_file"]
        if not frame_file.exists():
            continue
        frame = cv2.resize(_load_bgr(frame_file), (w, h),
                           interpolation=cv2.INTER_AREA)

        # Accumulate OF trace (already scaled)
        if idx in of_map:
            x, y, ok = of_map[idx]
            of_acc.append((int(round(x)), int(round(y)), ok))

        # Accumulate GT trace (already scaled)
        if idx in gt_map_scaled:
            gx, gy = gt_map_scaled[idx]
            gt_acc.append((int(round(gx)), int(round(gy))))

        # --- draw GT trace (green) ---
        vis_gt = gt_acc[-MAX_TRACE_TAIL:]
        for i in range(1, len(vis_gt)):
            cv2.line(frame, vis_gt[i - 1], vis_gt[i], GT_COLOR, TRACE_THICKNESS)
        if vis_gt:
            cv2.circle(frame, vis_gt[-1], POINT_RADIUS - 1, GT_COLOR, -1)

        # --- draw OF trace (yellow / dark-red if lost) ---
        vis_of = of_acc[-MAX_TRACE_TAIL:]
        for i in range(1, len(vis_of)):
            x0, y0, ok0 = vis_of[i - 1]
            x1, y1, ok1 = vis_of[i]
            colour = TRACE_COLOR if (ok0 and ok1) else LOST_COLOR
            cv2.line(frame, (x0, y0), (x1, y1), colour, TRACE_THICKNESS)
        if vis_of:
            cx, cy, c_ok = vis_of[-1]
            pt_col = POINT_COLOR if c_ok else (128, 128, 128)
            cv2.circle(frame, (cx, cy), POINT_RADIUS, pt_col, -1)
            cv2.circle(frame, (cx, cy), POINT_RADIUS + 2, (255, 255, 255), 1)

        # Seed marker
        if idx == seed_idx and of_acc:
            cv2.circle(frame, (of_acc[0][0], of_acc[0][1]), 12, SEED_COLOR, 2)

        # Drift text (compute in original pixel space)
        drift_txt = ""
        if idx in of_map and idx in gt_map_scaled:
            ox, oy, _ = of_map[idx]
            gx, gy = gt_map_scaled[idx]
            drift = float(np.hypot(ox - gx, oy - gy)) / scale  # back to orig px
            drift_txt = f" | drift={drift:.1f}px"

        # HUD
        fid = frames_list[idx]["frame_id"]
        cv2.putText(
            frame,
            f"frame_id={fid} (idx {idx}){drift_txt}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(frame, "Yellow: Optical Flow (LK)", (10, h - 60),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, TRACE_COLOR, 1)
        cv2.putText(frame, "Green: Tracking BBox Centre", (10, h - 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1)
        cv2.putText(frame, f"Seed: idx {seed_idx} (frame_id {frames_list[seed_idx]['frame_id']})",
                     (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SEED_COLOR, 1)

        writer.write(frame)

    writer.release()
    print(f"\nSaved → {output_path}  ({total} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load tracking manifest
    print(f"Loading {TRACKING_JSON}")
    with open(TRACKING_JSON) as f:
        manifest = json.load(f)
    frames = manifest["frames"]
    print(f"  {len(frames)} frames, fps inferred from frame count")

    # 2. Find most robust consecutive detected run
    run_start, run_len = find_longest_detected_run(frames)
    print(f"Longest detected run: idx {run_start}–{run_start + run_len - 1} "
          f"({run_len} frames)")

    # 3. Seed at the midpoint of that run
    seed_idx = run_start + run_len // 2
    seed_fr = frames[seed_idx]
    seed_x, seed_y = _bbox_centre(seed_fr["bbox"])
    print(f"Seed: idx={seed_idx}, frame_id={seed_fr['frame_id']}, "
          f"file={seed_fr['frame_file']}, pt=({seed_x:.1f}, {seed_y:.1f})")

    # 4. Track forward
    print("Tracking forward …")
    fwd = track_optical_flow(frames, seed_idx, (seed_x, seed_y), direction=1)
    print(f"  → {len(fwd)} frames")

    # 5. Track backward
    print("Tracking backward …")
    bwd = track_optical_flow(frames, seed_idx, (seed_x, seed_y), direction=-1)
    print(f"  → {len(bwd)} frames")

    # 6. Merge (backward reversed + forward, skip duplicate seed)
    full_trace = list(reversed(bwd)) + fwd[1:]
    print(f"Full OF trace: {len(full_trace)} frames")

    # 7. Build ground-truth map from tracking bbox centres
    gt_map: dict[int, tuple[float, float]] = {}
    for i, fr in enumerate(frames):
        if fr["detected"] and not fr["interpolated"]:
            gt_map[i] = _bbox_centre(fr["bbox"])

    # 8. Render video
    out_dir = OUTPUT_BASE / "optical_flow_precision" / VIDEO_STEM
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "optical_flow_trace.mp4"

    # Guess fps: the raw video is 30fps; extracted frames are 1:1 here
    fps = 30.0
    render_comparison_video(frames, full_trace, gt_map, out_path, fps, seed_idx)

    # 9. Dump trace data for further analysis
    trace_json = out_dir / "trace_data.json"
    drift_vals: list[float] = []
    for idx, ox, oy, ok in full_trace:
        if idx in gt_map:
            gx, gy = gt_map[idx]
            drift_vals.append(float(np.hypot(ox - gx, oy - gy)))

    trace_data = {
        "video_stem": VIDEO_STEM,
        "experiment_date": "2026-02-25",
        "seed_idx": seed_idx,
        "seed_frame_id": seed_fr["frame_id"],
        "seed_frame_file": seed_fr["frame_file"],
        "seed_point": [seed_x, seed_y],
        "lk_params": {
            "winSize": list(LK_PARAMS["winSize"]),
            "maxLevel": LK_PARAMS["maxLevel"],
        },
        "forward_frames": len(fwd),
        "backward_frames": len(bwd),
        "total_frames": len(full_trace),
        "drift_stats_vs_tracking_bbox": {
            "note": (
                "WARNING: drift is measured against tracking bbox centres which "
                "are WRONG in idx 448-545 (tracker follows wrong skier). "
                "High drift in that zone means OF is correct, not wrong."
            ),
            "mean_px": float(np.mean(drift_vals)) if drift_vals else None,
            "median_px": float(np.median(drift_vals)) if drift_vals else None,
            "max_px": float(np.max(drift_vals)) if drift_vals else None,
            "p90_px": float(np.percentile(drift_vals, 90)) if drift_vals else None,
        },
        "findings": {
            "tracker_identity_switch": {
                "description": (
                    "At idx 448-451 the tracker jumps 451px to a different skier "
                    "entering from frame-right. Tracker follows wrong person "
                    "through idx ~545. OF stays locked on Lach Powell."
                ),
                "switch_frame_idx": 450,
                "switch_frame_file": "frame_000600.jpg",
                "tracker_jump_px": 450.7,
                "of_position_at_switch": [968, 556],
                "tracker_position_at_switch": [1537, 529],
            },
            "of_smoothness": {
                "of_mean_displacement_per_frame": 4.56,
                "of_std_displacement": 2.71,
                "tracker_mean_displacement_per_frame": 6.01,
                "tracker_std_displacement": 17.92,
                "smoothness_ratio": "6.6x smoother (OF std / tracker std)",
            },
            "of_strengths": [
                "Maintains identity through detection switches",
                "Smoother trajectories (no detection noise / bbox jitter)",
                "Works well within ~150 frames of anchor point",
            ],
            "of_weaknesses": [
                "Accumulates drift beyond ~300 frames from seed",
                "Cannot recover from occlusion or scene change",
                "Tracks a single pixel patch, not the semantic object",
            ],
            "recommendation": (
                "Hybrid approach: use OF for short-range interpolation and as "
                "identity continuity validator. Re-anchor to detections every "
                "N frames. Use OF-predicted position to reject detection switches "
                "that are too far from the expected location."
            ),
        },
        "trace": [
            {"idx": t[0], "x": round(t[1], 2), "y": round(t[2], 2), "ok": t[3]}
            for t in full_trace
        ],
    }
    with open(trace_json, "w") as f:
        json.dump(trace_data, f, indent=2)

    print(f"\nDrift stats vs tracking bbox centres:")
    print("  (NOTE: high drift at idx 448-545 means OF is correct — tracker is on wrong skier)")
    for k, v in trace_data["drift_stats_vs_tracking_bbox"].items():
        if k == "note":
            continue
        print(f"  {k}: {v:.1f}" if v else f"  {k}: N/A")
    print(f"\nTrace data → {trace_json}")


if __name__ == "__main__":
    main()
