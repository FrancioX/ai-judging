# Experiments Tried

Everything that has been attempted across all experiment loops and manual runs. One-line outcome per entry.

---

## Tracking

### Phase A — Candidate Scoring & Conflict Resolution

| # | Experiment | Loop | Outcome | Result |
|---|-----------|------|:-------:|--------|
| Exp 1 | Velocity consistency scoring (w_velocity sweep) | loop-01 | Accepted (w=0.4, neutral) | Positive feedback loop when wrong candidate selected; committed at low weight for signal diversity only |
| Exp 2 | Synthetic OF candidates injected when no real detection nearby | loop-01 | **Accepted** | −17.7% mean error, +1.0% HOTA; became core of Phase A architecture |
| Exp 3 | Camera-motion-compensated velocity scoring | loop-01 | Rejected | No effect from CMC; regression was from higher w_velocity triggering feedback loops |
| Exp 5 | `identity_guard_max_jump_px` sweep (10/20/50/100/150) | loop-01 | **Accepted** | Lowered 150→20px; −0.6% mean error, +0.4% AssA via 77 additional true-positive rejections |
| Exp 9 | Boost track stickiness & continuity weights (no-OF mode) | loop-01 | Rejected | No change in decisions; candidate pool was bottleneck, not scoring weights |
| Exp 10 | Confidence-gated merge candidates (`min_detection_conf=0.6`) | loop-01 | Rejected | Too aggressive; removed valid low-confidence skier detections; +2.5px, −0.011 HOTA |
| Exp 14 | Diagnose Arno regression at `merge_score_threshold=0.6` | loop-01 | Accepted (diagnostic) | Identified single 10-detection fragment (track 70) lost at 0.6; 85.4px median GT distance |
| Loop13 Iter 5 | Phase A conflict weight sweep (`conflict_of_multiplier`, `conflict_stickiness_multiplier`) | loop-20260313 | **Accepted** | `of_mult 3.0→1.0`, `sticky_mult 0.2→1.0`; Quentin −2.6px, no regressions |
| Loop13 Iter 6 | Phase A conflict refinements (5 sub-approaches) | loop-20260313 | Rejected | All failed; Gabin Leonard has self-reinforcing wrong-seed feedback loop; needs ReID |
| Loop13 Iter 6b | Single-candidate-only OF re-anchoring | loop-20260313 | Rejected | Flat; seed selection itself is wrong, not re-anchoring policy |
| Loop13 Iter 6c | Longest-track seed for OF trace | loop-20260313 | Neutral (kept) | Flat but more principled than highest-score seed; retained |
| Loop13 Iter 6d | `conflict_of_multiplier=0.0` (eliminate OF from conflicts entirely) | loop-20260313 | Rejected | Gabin +2.9px; quality+continuity alone also picks bystander |
| Loop13 Iter 6e | Identity guard post-Phase-A (max_jump 150px/200px) | loop-20260313 | Rejected | Gabin +2.6px, Quentin +2.8px; aerial phases create legitimate large jumps |
| Loop13 Iter 6f | `conflict_velocity_multiplier` sweep (3.0, 5.0) | loop-20260313 | Rejected | Gabin +0.7–3.7px; self-reinforcing: velocity history also tracks bystander |
| Loop13 Iter 8 | Bbox area consistency scoring (`w_size` 0.3, 1.0) | loop-20260313 | Rejected | Null at all weights; bystanders are other athletes at similar distances with identical bbox areas |
| Loop13 Iter 9 | HSV color histogram appearance similarity (`w_color` 0.5, 2.0) | loop-20260313 | Rejected | Null; freeride bystanders wear similar gear; simple histograms provide no discriminative power |

### Phase A — Track Merging

| # | Experiment | Loop | Outcome | Result |
|---|-----------|------|:-------:|--------|
| Exp 8 | Raise `merge_score_threshold` (0.3→0.5) in no-OF mode | loop-01 | **Accepted** | Arno −36.8% error, +0.161 HOTA; best no-OF result at the time: 36.0px / 0.833 HOTA |
| Exp 11 | Sweep `merge_score_threshold` (0.45–0.70) | loop-01 | **Accepted** | `0.60` optimal; −30.8% error vs Exp 8 (24.9px), +0.022 HOTA on 18 videos |
| Exp 12 | Reintroduce OF on top of `merge_score_threshold=0.6` | loop-01 | Rejected | OF strongly degraded every video; gap-fill drifts onto bystanders when merge pool is smaller |
| Exp 15 | Hybrid adaptive threshold (0.5/0.6 based on detection-rate heuristic) | loop-01 | **Accepted** | Recovered Arno regression while preserving gains; 25.5px / 0.856 HOTA, 6.05ms/frame (speed-best) |

### Phase A — Architecture

| # | Experiment | Loop | Outcome | Result |
|---|-----------|------|:-------:|--------|
| Exp 4 | Phase A.6 jump-size instrumentation | loop-01 | Accepted (diagnostic) | Revealed bimodal jump distribution; enabled Exp 5 threshold sweep |
| Exp 13 | Candidate-pool diagnostics | loop-01 | Accepted (diagnostic) | Added `candidate_pool_stats` to manifests; enabled threshold tuning |
| Exp 19 | Decouple Phase A trace filtering from scoring (`of_trace_filter_enabled` flag) | loop-01 | Accepted (infrastructure) | Neutral metrics; unblocked safe re-testing of lightweight OF variants |

### Phase B — Gap Filling

| # | Experiment | Loop | Outcome | Result |
|---|-----------|------|:-------:|--------|
| Exp 16 | Lightweight OF: trace-only (no gap-fill) | loop-01 | Rejected | Catastrophic: +188–371px error; Phase A proximity gate over-filtered without gap-fill to re-anchor trace |
| Exp 17 | Lightweight OF: conditional gap-fill (`of_min_gap_for_fill=10`) | loop-01 | Rejected (infrastructure kept) | Neutral; conditions where OF helps weren't frequent enough |
| Exp 18 | Lightweight OF: high-confidence re-anchoring | loop-01 | Rejected | −0.178 HOTA regression; Phase A trace and Phase B gap-fill are inseparable |
| Exp 20 | OF trace as soft score (hard proximity filter removed) | loop-01 | **Accepted** | −7.8% error, +0.013 HOTA vs no-OF baseline; 23.5px / 0.869 HOTA (accuracy-best at the time) |
| Exp 21a | Conditional gap-fill stacked on Exp 20 (`of_min_gap_for_fill=10`, `of_drift_guard_px=200`) | loop-01 | **Accepted** | −1.1px, +0.007 HOTA, −3.3ms/frame; **22.4px / 0.876 HOTA, 41.9ms/frame (accuracy-best)** |
| Exp 21b | High-confidence re-anchoring stacked on Exp 20 | loop-01 | Rejected | Neutral |
| Loop13 Iter 1 | Sweep `of_min_gap_for_fill` (5/10/15/20) | loop-20260313 | Rejected | Completely flat; all problematic gaps in Arno/Quentin are >>20 frames (aerial phases) |
| Loop13 Iter 2 | Sweep `of_drift_guard_px` (50/100/150/200) | loop-20260313 | Rejected | Monotonically worse as tightened; OF is better than linear even during aerials |
| Loop13 Iter 3 | Camera motion compensation (ORB, ECC) | loop-20260313 | Rejected | +2.2px regression; background features latch onto sky/snow textures; ECC 2× slower for no gain |
| Loop13 Iter 4 | CoTracker3 for long-gap Phase B.1 fill (min_gap=20) | loop-20260313 | Rejected | Completely flat; all 10 tracked points fell back to LK (visibility <0.5 during aerials) |
| Loop14 Iter 10 | CoTracker3 with lower `min_visible_score: 0.2` | loop-20260314 | Rejected | CoTracker3 returns binary {0,1} visibility (internally thresholded at 0.5); `min_visible_score` below 0.5 has no effect |

### Phase C — Kalman Smoothing

| # | Experiment | Loop | Outcome | Result |
|---|-----------|------|:-------:|--------|
| Exp 6 | Kalman smoothing in no-OF mode (`smooth_window=5`) | loop-01 | **Accepted** | −1.8px, slightly faster; kept as Phase C foundation |
| Loop13 Iter 7 | Kalman segment re-init at long-gap boundaries (OC-SORT style) | loop-20260313 | Rejected | Null; error is in Phase B positions, not Kalman propagation; existing 10× R_interp/R_det already suppresses backward RTS |
| Loop13 Iter 10 | Kalman noise tuning (`r_interp_pos` 40→200) | loop-20260313 | Rejected | Completely flat; smoother is not the bottleneck; does not change gap-fill trajectory |
| Loop14 Iter 3 | `of_drift_guard_px: 400` (raised from 200→400) | loop-20260314 | **Accepted** | −11.1px mini-set; prevents premature OF→linear fallback |
| Loop14 Iter 12 | `flow_max_extrapolate_frames: 200` for trailing gaps | loop-20260314 | Rejected | Arno +5.6px; OF drifts badly in trailing invisible gaps; static copy is better |
| Loop14 Iter 13 | `of_min_gap_for_fill: 1000` (force linear interp for all internal gaps) | loop-20260314 | Rejected | +3.9px mean; bidirectional OF blending outperforms linear by 12.6px for Arno's 293-frame gap |
| Loop14 Iter 14 | `kalman_reinit_gap: 50` (OC-SORT style re-init) | loop-20260314 | Null | Completely flat; high r_interp_pos already prevents gap-fill corruption |
| Loop14 Iter 15 | `of_synthetic_confidence: 0.0` (disable synthetic OF candidates) | loop-20260314 | Null on mini-set | No conflicts during large aerial gaps; untested on Gabin Leonard bystander-lock |

### No-OF / Speed Track

| # | Experiment | Loop | Outcome | Result |
|---|-----------|------|:-------:|--------|
| Exp 7 | No-OF identity guard using previous detection | loop-01 | Rejected | Strong regression: +108.9px mean error, −0.259 HOTA |

---

## Segmentation

| # | Experiment | Loop | Outcome | Result |
|---|-----------|------|:-------:|--------|
| Loop14 Iters 4–7 | `segmentation.confidence: 0.3` (lowered from 0.5) | loop-20260314 | **Accepted** | −13.2px dev-set (50% reduction); Jordan Koch 42.6→9.9px, Theodor Salen 54.6→9.2px, Quentin 47.8→13.6px |
| Loop14 Iter 8 | `select_strategy: "largest"` vs `"center"` | loop-20260314 | Null | Identical at conf=0.3; single largest detection per frame is same as highest-confidence selection |
| Loop14 Iter 11 | `imgsz: 1920` for YOLO11x-seg | loop-20260314 | Failed (MPS) | `NotImplementedError: Output channels > 65536` on Apple Silicon; requires CUDA or CPU (20–30 min/video) |

---

## Best Results Achieved

| Target | Config | Mean err (px) | HOTA | Speed (ms/frame) |
|--------|--------|:---:|:---:|:---:|
| **Accuracy-best** | Loop14 (conf=0.3, drift_guard=400) | 13.0 (dev-10) | 0.924 | ~45.0 |
| **Speed-best** | Exp 15 (adaptive threshold, no OF) | 25.5 (18-video) | 0.856 | 6.05 |
