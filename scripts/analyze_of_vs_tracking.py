"""Analyse optical-flow trace vs tracking bbox centres.

Focuses on finding where the tracker jumps to wrong person while OF stays correct.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

VIDEO_STEM = "VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86"
OF_JSON = Path(f"output/optical_flow_precision/{VIDEO_STEM}/trace_data.json")
TK_JSON = Path(f"output/tracking/{VIDEO_STEM}/tracking.json")

with open(OF_JSON) as f:
    td = json.load(f)
with open(TK_JSON) as f:
    tk = json.load(f)

frames = tk["frames"]
trace = td["trace"]
seed_idx = td["seed_idx"]

of_map: dict[int, tuple[float, float, bool]] = {
    t["idx"]: (t["x"], t["y"], t["ok"]) for t in trace
}

# ── 1. Full tracking bbox jumps ──────────────────────────────────────────
print("=" * 80)
print("1. ALL tracking bbox jumps > 30px (frame-to-frame)")
print("=" * 80)
prev_cx = prev_cy = None
for i, fr in enumerate(frames):
    cx = (fr["bbox"][0] + fr["bbox"][2]) / 2
    cy = (fr["bbox"][1] + fr["bbox"][3]) / 2
    if prev_cx is not None:
        d = math.hypot(cx - prev_cx, cy - prev_cy)
        if d > 30:
            of_info = ""
            if i in of_map:
                ox, oy, ok = of_map[i]
                of_info = f"  OF=({ox:.0f},{oy:.0f})"
            print(
                f"  idx {i-1:>4}→{i:<4}  jump={d:>6.1f}px  "
                f"trk=({cx:.0f},{cy:.0f})  interp={fr['interpolated']}"
                f"  track_id={fr['track_id']}{of_info}"
            )
    prev_cx, prev_cy = cx, cy

# ── 2. Detailed view around idx 400-550 ──────────────────────────────────
print()
print("=" * 80)
print("2. Detail: idx 400-550 — tracking bbox vs OF position")
print("=" * 80)
header = (
    f"{'idx':>5} {'frame_file':>18} {'trk_cx':>7} {'trk_cy':>7} "
    f"{'of_x':>7} {'of_y':>7} {'jump':>7} {'interp':>6} {'tid':>4}"
)
print(header)
print("-" * len(header))
prev_cx = prev_cy = None
for i in range(400, min(550, len(frames))):
    fr = frames[i]
    cx = (fr["bbox"][0] + fr["bbox"][2]) / 2
    cy = (fr["bbox"][1] + fr["bbox"][3]) / 2
    jump_s = ""
    if prev_cx is not None:
        d = math.hypot(cx - prev_cx, cy - prev_cy)
        if d > 15:
            jump_s = f"{d:.0f}"
    prev_cx, prev_cy = cx, cy
    of_x = of_y = ""
    if i in of_map:
        of_x = f"{of_map[i][0]:.0f}"
        of_y = f"{of_map[i][1]:.0f}"
    print(
        f"{i:>5} {fr['frame_file']:>18} {cx:>7.0f} {cy:>7.0f} "
        f"{of_x:>7} {of_y:>7} {jump_s:>7} {str(fr['interpolated']):>6} "
        f"{fr['track_id']:>4}"
    )

# ── 3. OF smoothness vs tracking smoothness ──────────────────────────────
print()
print("=" * 80)
print("3. Frame-to-frame displacement (smoothness)")
print("=" * 80)
of_indices = sorted(of_map.keys())
of_disps = []
tk_disps = []
for idx in of_indices[1:]:
    prev = idx - 1
    if prev in of_map and idx in of_map:
        ox0, oy0, _ = of_map[prev]
        ox1, oy1, _ = of_map[idx]
        of_disps.append(math.hypot(ox1 - ox0, oy1 - oy0))
    if prev < len(frames) and idx < len(frames):
        fr0, fr1 = frames[prev], frames[idx]
        gx0 = (fr0["bbox"][0] + fr0["bbox"][2]) / 2
        gy0 = (fr0["bbox"][1] + fr0["bbox"][3]) / 2
        gx1 = (fr1["bbox"][0] + fr1["bbox"][2]) / 2
        gy1 = (fr1["bbox"][1] + fr1["bbox"][3]) / 2
        tk_disps.append(math.hypot(gx1 - gx0, gy1 - gy0))

print(f"  OF:       mean={np.mean(of_disps):.2f}  std={np.std(of_disps):.2f}  "
      f"max={np.max(of_disps):.1f}")
print(f"  Tracking: mean={np.mean(tk_disps):.2f}  std={np.std(tk_disps):.2f}  "
      f"max={np.max(tk_disps):.1f}")

# ── 4. Identify phases ───────────────────────────────────────────────────
# Phase analysis: split into regions where OF-to-tracking distance changes
print()
print("=" * 80)
print("4. Phase analysis — OF vs tracking drift by 50-frame windows")
print("=" * 80)
chunk = 50
print(f"{'Window':>12} {'mean_drift':>10} {'median':>8} {'max':>8} "
      f"{'n_interp':>8} {'n_jumps>30':>10}")
for start in range(of_indices[0], of_indices[-1] + 1, chunk):
    end = min(start + chunk, of_indices[-1] + 1)
    drifts = []
    n_interp = 0
    n_jumps = 0
    for idx in range(start, end):
        if idx in of_map and idx < len(frames):
            ox, oy, ok = of_map[idx]
            fr = frames[idx]
            gx = (fr["bbox"][0] + fr["bbox"][2]) / 2
            gy = (fr["bbox"][1] + fr["bbox"][3]) / 2
            drifts.append(math.hypot(ox - gx, oy - gy))
            if fr["interpolated"]:
                n_interp += 1
        if idx > start and idx < len(frames):
            fr0, fr1 = frames[idx - 1], frames[idx]
            cx0 = (fr0["bbox"][0] + fr0["bbox"][2]) / 2
            cy0 = (fr0["bbox"][1] + fr0["bbox"][3]) / 2
            cx1 = (fr1["bbox"][0] + fr1["bbox"][2]) / 2
            cy1 = (fr1["bbox"][1] + fr1["bbox"][3]) / 2
            if math.hypot(cx1 - cx0, cy1 - cy0) > 30:
                n_jumps += 1
    if drifts:
        label = f"{start}-{end - 1}"
        print(
            f"{label:>12} {np.mean(drifts):>10.1f} {np.median(drifts):>8.1f} "
            f"{np.max(drifts):>8.1f} {n_interp:>8} {n_jumps:>10}"
        )

# ── 5. Summary stats ─────────────────────────────────────────────────────
print()
print("=" * 80)
print("5. Summary")
print("=" * 80)
print(f"  Seed: idx={seed_idx}, frame={frames[seed_idx]['frame_file']}")
print(f"  OF trace: {len(trace)} frames (idx {of_indices[0]}-{of_indices[-1]})")
print(f"  Forward: {td['forward_frames']} frames, Backward: {td['backward_frames']} frames")
print(f"  Total tracking frames: {len(frames)}")

all_drifts = []
for t in trace:
    idx = t["idx"]
    if idx < len(frames):
        fr = frames[idx]
        gx = (fr["bbox"][0] + fr["bbox"][2]) / 2
        gy = (fr["bbox"][1] + fr["bbox"][3]) / 2
        all_drifts.append(math.hypot(t["x"] - gx, t["y"] - gy))
all_drifts_np = np.array(all_drifts)
print(f"  Overall drift: mean={np.mean(all_drifts_np):.1f}px  "
      f"median={np.median(all_drifts_np):.1f}px  "
      f"p90={np.percentile(all_drifts_np, 90):.1f}px  "
      f"max={np.max(all_drifts_np):.1f}px")

n_interp_total = sum(1 for fr in frames if fr["interpolated"])
print(f"  Interpolated frames in tracking: {n_interp_total}/{len(frames)} "
      f"({100*n_interp_total/len(frames):.1f}%)")
