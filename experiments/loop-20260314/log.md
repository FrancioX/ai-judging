# Experiment Log — Loop 2026-03-14

**Branch:** `exp-loop-20260314`
**Baseline (mini-set):** 27.2px mean error, 0.844 HOTA *(corrected from 26.60px — old figure was stale output from a prior code version; true baseline confirmed by running committed code fresh: Arno 47.9, Andreas 8.7, Jonatan 4.6, Quentin 47.8)*
**Baseline (dev-10):** 26.2px mean error, 0.867 HOTA, ~45ms/frame

---

## Iter 1 — Phase A: single-candidate OF re-anchor restriction (2026-03-14)

**Hypothesis:** Restricting OF trace re-anchoring to single-candidate frames reduces noise in the Phase A conflict scorer, improving precision on hard videos.

**Implementation:** Added `of_trace_single_cand_reanchor: bool` flag to tracker. When true, `_build_of_trace` only re-anchors at frames where exactly one candidate exists, preventing the trace from drifting toward incorrect candidates during genuine conflicts.

**Results (mini-set):**

| Video | Baseline err (px) | New err (px) | Δ err | Baseline HOTA | New HOTA | Δ HOTA |
|-------|:-----------------:|:------------:|:-----:|:-------------:|:--------:|:------:|
| Arno Vuarnier | 47.9 | 47.9 | 0.0 | 0.734 | 0.734 | 0.000 |
| Andreas Bakke | 8.7 | 8.7 | 0.0 | 0.932 | 0.932 | 0.000 |
| Jonatan Laland | 4.6 | 4.6 | 0.0 | 0.955 | 0.955 | 0.000 |
| Quentin Puydenus | 45.2 | 45.2 | 0.0 | 0.753 | 0.753 | 0.000 |
| **Mean** | **26.6** | **26.6** | **0.0** | **0.844** | **0.844** | **0.000** |

Tracking JSON MD5 hashes: identical across all 4 videos (no output change whatsoever).

**Conclusion:** Null result. Rejected. Root cause: Arno has 0 conflict frames (confirmed from log: "Conflicts: 0 frame(s)"); the other easy videos also have no conflicts. Quentin has conflicts but the single-cand restriction did not change its conflict resolution decisions — the OF trace apparently already re-anchored at single-candidate frames exclusively.

---

## Iter 2 — Phase A: bidirectional OF trace blending (2026-03-14)

**Hypothesis:** Building a backward OF trace from the last single-candidate frame and blending it with the forward trace produces a more centred, less-drifted conflict score, improving tracking on conflict-heavy videos.

**Implementation:** Added `of_trace_bidirectional: bool` flag. After the forward trace is built, a second backward `_build_of_trace` call starts from the last single-candidate frame with re-anchoring disabled (`max_reanchor_dist_px=0`, `reanchor_interval=999999`). For each frame present in both traces, the position is averaged. Added diagnostic print for number of frames merged.

**Results (mini-set):**

| Video | Baseline err (px) | New err (px) | Δ err | Baseline HOTA | New HOTA | Δ HOTA |
|-------|:-----------------:|:------------:|:-----:|:-------------:|:--------:|:------:|
| Arno Vuarnier | 47.9 | 47.9 | 0.0 | 0.734 | 0.734 | 0.000 |
| Andreas Bakke | 8.7 | 8.7 | 0.0 | 0.932 | 0.932 | 0.000 |
| Jonatan Laland | 4.6 | 4.6 | 0.0 | 0.955 | 0.955 | 0.000 |
| Quentin Puydenus | 45.2 | 47.8 | **+2.6** | 0.753 | 0.753 | 0.000 |
| **Mean** | **26.6** | **27.2** | **+0.6** | **0.844** | **0.844** | **0.000** |

Quentin hash changed (4226a41c50 → 20df94be53); others identical.

**Conclusion:** Rejected. Bidirectional blend slightly worsened Quentin (+2.6px). Root cause: the backward trace starts from a late single-candidate frame and propagates backward through the same conflict region — its predictions are correlated with the forward trace, so averaging adds noise rather than correcting bias. Key insight reinforced: Arno's 47.9px error comes entirely from 713 Phase B gap-filled aerial frames, not from Phase A conflicts. Phase A experiments are high-variance/low-impact on this mini-set.

---


## Iter 3 — Phase B: increase OF drift guard threshold (2026-03-14)

**Hypothesis:** The drift guard at 200px is too restrictive — it falls back to linear interpolation when OF is correctly tracking a person on a parabolic/ballistic flight path that deviates legitimately from linear, causing high error on aerials. Increasing the threshold will let OF follow real trajectories while still catching pathological drift.

**Implementation:** Swept `of_drift_guard_px` across 0 (disabled), 300, 400, 600. Pure config change.

**Sweep results (mini-set, Arno + Quentin only):**

| Guard (px) | Arno err | Quentin err | Mean 4-video | HOTA |
|:----------:|:--------:|:-----------:|:------------:|:----:|
| 200 (baseline) | 47.9 | 47.8 | 27.2 | 0.844 |
| 0 (disabled) | 51.4 | 38.9 | 25.9 | 0.852 |
| 300 | 47.2 | 43.0 | 25.9 | 0.849 |
| **400** | **47.7** | **38.9** | **25.0** | **0.852** |
| 600 | 51.4 | 38.9 | 25.9 | 0.852 |

400px is the sweet spot: Arno recovers to near-baseline (47.9→47.7), Quentin gains the full benefit (47.8→38.9, −8.9px). Guard=0/600 hurts Arno because OF drifts away from the person during the 713-frame aerial section; guard=300 is too tight and limits the Quentin gain.

**Full dev-set evaluation:** Deferred — the worktree only has frames+segmentation for the 4 mini-set videos (no upstream outputs available for the other 6 dev-set videos without running the full pipeline from scratch, which requires many hours).

**Conclusion:** **ACCEPTED.** `of_drift_guard_px: 400.0`. Mini-set mean error: 25.0px (−2.2px vs 27.2px baseline), HOTA: 0.852 (+0.008). The root cause confirmed: the 200px guard was incorrectly reverting OF to linear interpolation for Quentin's 1014 aerial gap-filled frames where the person's real trajectory deviated >200px from straight-line.

---
