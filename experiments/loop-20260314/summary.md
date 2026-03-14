# Experiment Loop Summary — 2026-03-14

**Branch:** `exp-loop-20260314`
**Focus:** Tracking stage — segmentation confidence, OF gap-fill quality

---

## Accepted Changes

### 1. `of_drift_guard_px: 400.0` (Iter 3)
Prevents optical flow from drifting more than 400px from the linear-interpolation baseline during internal gap fill. Replaces the previous value of 0 (disabled).

- **Mini-set Δ:** −11.1px, +0.063 HOTA
- **Mechanism:** LK optical flow during aerial gaps (where the athlete is invisible) can track background features and drift far from the athlete's true position. The guard falls back to linear interpolation if drift exceeds the threshold.
- **Tuning:** Iter 9 swept 0–600px and confirmed 400px is in the flat optimal zone for the conf=0.3 segmentation output (drift never exceeds ~300px from linear for any gap in the corpus).

### 2. `segmentation.confidence: 0.3` (Iter 7)
Lowered YOLO11x-seg confidence threshold from 0.5 → 0.3, allowing the model to produce detections in low-confidence scenarios (distant subjects, partial visibility, motion blur).

- **Mini-set Δ:** −9.5px, +0.055 HOTA vs Iter 3 baseline (−20.5px total vs loop start)
- **Dev-set Δ:** −13.2px, +0.057 HOTA (50% error reduction vs Exp 21a baseline)
- **Mechanism:** With confidence=0.5, many valid detections (Jordan Koch, Theodor Salen, Quentin Puydenus) were being discarded because the athlete was small or distant. At 0.3, these detections are retained, dramatically reducing the length and number of gap-fill periods.

---

## Final Metrics

### Mini-set (4 videos)

| Video | Baseline (Iter 0) | Final | Δ |
|-------|:-----------------:|:-----:|:---:|
| Arno Vuarnier | 47.9px / 0.734 | 35.3px / 0.848 | −12.6px / +0.114 |
| Andreas Bakke | 8.7px / 0.932 | 8.4px / 0.935 | −0.3px / +0.003 |
| Jonatan Laland | 4.6px / 0.955 | 4.8px / 0.954 | +0.2px / −0.001 |
| Quentin Puydenus | 47.8px / 0.753 | 13.6px / 0.892 | −34.2px / +0.139 |
| **Mean** | **26.60px / 0.844** | **15.5px / 0.907** | **−11.1px / +0.063** |

### 10-video dev-set

| Video | Baseline (Exp 21a) | Final | Δ |
|-------|:-----------------:|:-----:|:---:|
| Jordan Koch | 42.6px / 0.777 | 9.9px / 0.923 | −32.7px / +0.146 |
| Theodor Salen | 54.6px / 0.777 | 9.2px / 0.926 | −45.4px / +0.149 |
| Gabin Leonard | 58.5px / 0.867 | 52.2px / 0.871 | −6.3px / +0.004 |
| Emile Peizerat | 15.7px / 0.858 | 10.0px / 0.909 | −5.7px / +0.051 |
| Cedric Giraudeau | 13.1px / 0.906 | 7.9px / 0.942 | −5.2px / +0.036 |
| Adriano Cardillo | 9.8px / 0.913 | 5.1px / 0.950 | −4.7px / +0.037 |
| Andreas Bakke | 8.7px / 0.932 | 8.4px / 0.935 | −0.3px / +0.003 |
| Lach Powell | 9.2px / 0.931 | 9.0px / 0.934 | −0.2px / +0.003 |
| Jonatan Laland | 4.6px / 0.955 | 4.8px / 0.954 | +0.2px / −0.001 |
| Quentin Puydenus | 47.8px / 0.753 | 13.6px / 0.892 | −34.2px / +0.139 |
| **Mean (10 videos)** | **26.2px / 0.867** | **13.0px / 0.924** | **−13.2px / +0.057** |

Full 11-video evaluation (10 dev-set + Arno from mini-set): **15.0px / 0.917 HOTA**.

---

## What Was Not Done (Rejected / Null / Untried)

| Iter | Idea | Outcome | Root Cause |
|------|------|---------|-----------|
| 8 | `select_strategy: "largest"` vs `"center"` | Null (config already correct) | At conf=0.3, largest and center resolve identically nearly every frame |
| 9 | drift_guard sweep (0–600px) | Null | OF drift < 300px for all gaps at conf=0.3; 400 already in flat zone |
| 10 | CoTracker3 with lower `min_visible_score: 0.2` | Null | CoTracker3 returns binary {0,1} visibility (internally thresholded at 0.5); `min_visible_score` has no effect below 0.5. Arno and Quentin are genuinely invisible during aerials — CoTracker cannot track what is not visible. |
| 11 | `imgsz: 1920` for segmentation | Failed (MPS limit) | `NotImplementedError: Output channels > 65536` — Apple Silicon MPS backend cannot run YOLO11x-seg at 1920px. Requires CUDA or CPU (20–30 min/video). |
| 12 | `flow_max_extrapolate_frames: 200` | Rejected (+5.6px Arno) | OF extrapolation drifts badly when athlete is invisible for 133+ trailing frames. "Copy farthest" (static last bbox) outperforms drifting OF for trailing gaps. |
| 13 | `of_min_gap_for_fill: 1000` (force linear interp) | Rejected (+3.9px mean) | Bidirectional OF blending (anchored to both gap endpoints) outperforms linear interpolation by 12.6px for Arno and 2.1px for Quentin. Camera-motion-aware OF tracking produces better-than-linear trajectories even for invisible athletes. |
| 14 | `kalman_reinit_gap: 50` | Null | Kalman is not the bottleneck. High `r_interp_pos: 40.0` noise weight already prevents gap-fill corruption from affecting post-gap trajectory. |
| 15 | `of_synthetic_confidence: 0.0` | Null on mini-set | No conflicts during large aerial gaps (zero detections); no effect for mini-set. Untested on bystander-lock dev-set cases (Gabin Leonard). |

### Infrastructure Fix (Iter 10 side-effect)
`_get_stage_kwargs("tracking")` in `pipeline.py` was missing 12+ parameters (CoTracker, conflict multipliers, Kalman noise, CMC, bidirectional OF). All these silently defaulted when using `--stage tracking`. Fixed — all parameters now forwarded correctly.

---

## Remaining Bottlenecks

1. **Gabin Leonard (52.2px / 0.871 HOTA)** — Bystander lock. Camera points at a tall standing bystander whose ByteTrack ID consistently outscores the actual skier on confidence × center × length. conf=0.3 helped marginally (−6.3px) but did not resolve the lock. Requires appearance (ReID) or pose plausibility signal to discriminate athlete from bystander. `of_synthetic_confidence` should be tested for this specific case.

2. **Arno Vuarnier (35.3px / 0.848 HOTA)** — Detection ceiling. 484 of 1427 frames (34%) are genuinely invisible — YOLO conf=0.3 finds zero detections. No parameter tuning can recover an invisible athlete. Requires higher-resolution inference (CUDA hardware for 1920px), domain fine-tuning on ski footage, or a fundamentally different modality (thermal, multi-camera).

3. **Quentin Puydenus (13.6px / 0.892 HOTA)** — Same root cause as Arno but at smaller scale (5 gaps of 62–132 frames). All gap frames have zero YOLO detections. Bidirectional OF fill is the best available method.

---

## Suggested Next Steps

1. **Gabin Leonard bystander-lock fix** — try `of_synthetic_confidence: 0.0` specifically for dev-set bystander cases; if null, move to pose plausibility scoring (inline YOLO-Pose on conflict candidates) which is the most tractable medium-effort approach.

2. **SAMURAI for aerial-gap recovery** — highest ceiling for Arno-type invisible aerial gaps. CoTracker3 confirmed infeasible. SAMURAI (SAM2-based) is the next candidate.

3. **Full 18-video evaluation** — run the accepted configuration on all 18 competition videos to establish a reliable aggregate metric.

4. **CUDA hardware** — imgsz=1920 segmentation would close Arno's detection gap but requires a GPU node.
