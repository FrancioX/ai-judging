# Experiment Loop Summary — 2026-03-13/14

**Branch:** `exp-loop-20260313`
**Focus:** Tracking stage improvements
**Duration:** 10 iterations over two days

---

## Baseline

| State | Mean err (px) | HOTA | Speed (ms/frame) |
|-------|:---:|:---:|:---:|
| Exp 21a (loop start) | 22.4 | 0.876 | ~41.9 |
| Mini-set start | 27.25 | 0.844 | ~45.9 |

**Primary failure cases at loop start:** Arno Vuarnier (47.9 px), Quentin Puydenus (47.8 px), Gabin Leonard (58.5 px), Theodor Salen (54.6 px)

---

## Final Result

**One accepted change** — Phase A conflict-resolution weight rebalancing (Iteration 5).

| State | Mean err (px) | HOTA |
|-------|:---:|:---:|
| Mini-set (4 videos) | 26.60 | 0.844 |
| Dev-set (10 videos) | 26.2 | 0.867 |

Net improvement on dev set: **−0.3 px mean error**, all from Quentin Puydenus (−2.6 px). No regressions.

---

## Iteration Results

| # | Idea | Outcome | Δ mean err (mini) |
|---|------|:-------:|:-----------------:|
| 1 | Sweep `of_min_gap_for_fill` (5/10/15/20) | Rejected | 0.0 |
| 2 | Sweep `of_drift_guard_px` (50–200) | Rejected | 0.0 (tightening is worse) |
| 3 | Camera motion compensation (ORB, ECC) | Rejected | +2.2 |
| 4 | CoTracker3 for long-gap Phase B fill | Rejected | 0.0 |
| **5** | **Phase A conflict weights (`of_mult=1.0`, `sticky_mult=1.0`)** | **Accepted** | **−0.65** |
| 6 | Further Phase A conflict refinements (5 sub-approaches) | All rejected | 0.0 to +2.9 |
| 7 | Kalman segment re-init at gap boundaries (OC-SORT style) | Rejected | 0.0 |
| 8 | Bbox area consistency scoring (`w_size`) | Rejected | 0.0 |
| 9 | HSV color histogram appearance scoring (`w_color`) | Rejected | 0.0 |
| 10 | Kalman noise tuning (`r_interp_pos=200`) | Rejected | 0.0 |

---

## Key Findings

### What worked
- **Rebalancing Phase A conflict weights** (`conflict_of_multiplier: 3.0 → 1.0`, `conflict_stickiness_multiplier: 0.2 → 1.0`): reducing the OF trace's outsized influence during candidate conflicts allowed track stickiness to contribute more equally, resolving a bystander lock-on case in Quentin Puydenus.

### What didn't work — and why

**Gap-fill parameter tuning is at its ceiling.** Arno Vuarnier has 713 interpolated frames out of ~1400 (50% gap fill from aerial phases). Every gap-fill improvement tried — threshold tuning, drift guard sweep, CoTracker3, camera motion compensation — produced flat or worse results. The error during aerial phases is irreducible with the current OF-based approach: the skier is genuinely undetectable and LK/Farneback track background features, not the skier.

**Kalman smoothing is not the bottleneck.** Experiments 7 and 10 confirmed that the gap-fill positions themselves are wrong — the smoother is downstream and cannot correct upstream Phase B errors. The existing 10× noise ratio (`R_interp/R_det`) already largely suppresses backward RTS propagation across gap boundaries.

**Phase A bystander disambiguation requires learned appearance embeddings.** Gabin Leonard's failure is a self-reinforcing feedback loop: a wrong OF trace seed → wrong conflict decisions → wrong velocity history → more wrong conflict decisions. Simple signals tried — color histograms (Iter 9), bbox area (Iter 8), velocity consistency boost (Iter 6), identity guard (Iter 6) — all failed because bystanders in freeride competitions are other athletes wearing similar gear at similar distances. Only a trained ReID or pose-plausibility model could break this loop.

**CMC degrades performance.** Camera motion compensation (ORB and ECC) causes regressions because background features in aerial phases latch onto sky/snow textures, degrading the OF estimate rather than isolating camera motion.

---

## Infrastructure Added (retained despite rejection)

These components were implemented, wired through config, and kept at their default (disabled) values for future use:

| Component | Location | Config key | Notes |
|-----------|----------|------------|-------|
| CoTracker3 fill | `src/tracking/cotracker_fill.py` | `cotracker_enabled: false` | Falls back to LK; needs lower `min_visible` threshold or SAMURAI to be useful |
| Conflict velocity multiplier | `tracker.py` | `conflict_velocity_multiplier: 1.0` | Wired but ineffective alone; may help once ReID breaks the feedback loop |
| Color histogram scorer | `tracker.py` | `w_color: 0.0` | Zero cost when disabled; needs ReID embeddings |
| Bbox area scorer | `tracker.py` | `w_size: 0.0` | Zero cost when disabled; ineffective in freeride domain |
| Kalman gap re-init | `tracker.py` | `kalman_reinit_gap: 0` | Zero cost; may become relevant if gap-fill quality improves |
| Longest-track OF seed | `tracker.py` | (always on) | More principled than highest-score seed; kept neutral |

---

## Recommended Next Steps

The tracking stage has hit the limit of what is achievable without learning-based appearance models. To advance further:

1. **OSNet / FastReID** — train or fine-tune a ReID model on ski/snowboard competition footage. Even approximate color+silhouette embeddings would break the Gabin Leonard feedback loop.
2. **SAMURAI** — SAM2-based zero-shot tracking for aerial gap recovery. Identified in the original candidate list as the highest-ceiling, longest-effort option; the CoTracker infrastructure is a stepping stone.
3. **Pose plausibility** — inline YOLO-Pose on Phase A candidates to filter bystanders who are standing still or facing away.

The easy wins (parameter sweeps, lightweight OF improvements) are exhausted. Any meaningful improvement requires one of the above.
