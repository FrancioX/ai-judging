# Optical Flow Tracking Precision — Experiment Results

**Date**: 2026-02-25
**Video**: Lach Powell — VERBIER FREERIDE WEEK QUALIFIER 4, Heat 3, Ski Men
**File**: `VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86.mp4`

---

## Setup

- **Method**: Lucas-Kanade optical flow (pyramidal, `winSize=21×21`, `maxLevel=3`, 30 iterations, eps=0.01)
- **Seed point**: bbox centre (646, 285) at idx 319 (`frame_000469.jpg`) — midpoint of the longest consecutive detected run (idx 99–539, 441 frames)
- **Tracking direction**: forward 477 frames + backward 320 frames = **796 total frames**
- **Comparison baseline**: detection-based tracker (YOLO segmentation + track merging + optical-flow-assisted interpolation)

## Output

- Video: `output/optical_flow_precision/.../optical_flow_trace.mp4` (yellow = OF, green = tracker bbox centres)
- Data: `output/optical_flow_precision/.../trace_data.json`
- Analysis script: `scripts/analyze_of_vs_tracking.py`

---

## Key Finding: OF Prevents Identity Switch

At **idx 448–451** (frames 598–601, ~15s mark), the detection-based tracker **jumps 451px to a different skier** entering from the right side of the frame:

| Frame idx | Tracker position | OF position | What happened |
|-----------|-----------------|-------------|---------------|
| 447 | (985, 545) | (965, 548) | Normal — both on Lach |
| 448 | (1011, 546) | (966, 550) | Tracker starts drifting |
| 449 | (1086, 544) | (967, 553) | Tracker jumps 76px |
| **450** | **(1537, 529)** | **(968, 556)** | **Tracker jumps 451px to wrong skier** |
| 451 | (1616, 524) | (970, 558) | Tracker follows wrong person |

The tracker follows the wrong person from idx 450 through ~545 (~3.2 seconds). **Optical flow stays locked on Lach Powell** throughout, with only ~2px frame-to-frame movement — typical for smooth skiing motion.

---

## Phase-by-Phase Analysis

| Window (idx) | OF↔Tracker drift | Interp frames | Tracker jumps>30px | Interpretation |
|-------------|-------------------|---------------|-------------------|----------------|
| 0–49 | 121 px | 0 | 0 | OF backward drift from seed |
| 50–99 | 33 px | 1 | 0 | Moderate backward drift |
| 100–199 | 12–21 px | 0 | 0 | Close to seed, good agreement |
| 200–349 | **3–7 px** | 0 | 0 | **Seed zone — near-perfect match** |
| 350–449 | 16–19 px | 0 | 1 | Slight divergence, tracker starting to drift |
| **450–549** | **459–668 px** | 7 | 10 | **Tracker on wrong skier; OF is correct** |
| 550–699 | 29–38 px | 38 | 0 | Tracker recovers; moderate OF drift |
| 700–795 | 90–506 px | 60 | 0 | OF accumulates drift; heavy interpolation |

---

## Smoothness Comparison

| Metric | Optical Flow | Detection Tracker |
|--------|-------------|-------------------|
| Mean frame-to-frame displacement | 4.56 px | 6.01 px |
| Std frame-to-frame displacement | **2.71 px** | **17.92 px** |
| Max frame-to-frame displacement | 29.2 px | 450.7 px |

OF produces **6.6× smoother** trajectories (by std). The tracker's high variance comes from detection bbox jitter and the identity switch.

---

## Conclusions

### OF Strengths
1. **Identity continuity** — tracks pixel appearance, immune to detection-level ID switches
2. **Smoothness** — no bbox jitter, no detection noise, no sudden jumps
3. **Reliable short-range** — excellent within ±150 frames (~5s) of anchor point

### OF Weaknesses
1. **Cumulative drift** — unreliable beyond ~300 frames without re-anchoring
2. **No recovery** — once lost (occlusion, fast motion blur), cannot re-acquire
3. **Pixel-level, not semantic** — tracks a patch, not "the skier"

### Recommendation: Hybrid Approach

Use optical flow as an **identity-switch guard** in the existing tracker:

1. Maintain an OF-predicted position alongside detection-based tracking
2. When the tracker proposes a large bbox jump (>N px), compare against OF prediction
3. If the new detection is far from OF prediction but close to a *different* person, **reject the switch**
4. Re-anchor OF to the detection bbox centre every M frames where detections are confident
5. Use OF for short-range gap interpolation (currently done, but not for identity validation)

This would have prevented the idx 450 identity switch: OF predicted ~(968, 556) while the tracker wanted to jump to (1537, 529) — a 569px discrepancy that should trigger rejection.
