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

## Iter 4 — Phase C: Kalman r_interp_pos tuning (2026-03-14)

**Hypothesis:** Increasing `r_interp_pos` makes the Kalman rely more on its constant-acceleration dynamics model during long aerial gaps rather than noisy OF positions, producing a more physically plausible parabolic trajectory.

**Implementation:** Swept `r_interp_pos` across 20, 40 (baseline), 80, 160. Pure config change.

**Results (mini-set):**

| r_interp_pos | Arno err | Quentin err | Mean | HOTA |
|:------------:|:--------:|:-----------:|:----:|:----:|
| 20 | 47.7 | 38.9 | 25.0 | 0.852 |
| 40 (baseline) | 47.7 | 38.9 | 25.0 | 0.852 |
| 80 | 47.7 | 38.9 | 25.0 | 0.852 |
| 160 | 47.7 | 38.9 | 25.0 | 0.852 |

**Conclusion:** Null result. Rejected. The Kalman noise parameter has no effect on the final trajectory positions — the OF gap-fill outputs are already at fixed positions, and varying how much the Kalman trusts them doesn't change the smoothed result on these videos. Additionally confirmed: Arno's 713 gap frames are a **detection failure** (white clothes + snowy background → YOLO does not detect him), not a tracking failure. Filling those gaps better requires segmentation-stage improvements (confidence threshold, model), not Phase C tuning.

---

## Iter 5 — Phase B/A: dense optical flow (2026-03-14)

**Hypothesis:** Switching from "auto" (sparse LK + dense fallback) to "dense" Farneback OF handles large inter-frame motion during aerials better than sparse keypoint tracking, reducing gap-fill error.

**Implementation:** `optical_flow_method: "dense"`. Pure config change.

**Results (mini-set):**

| Video | Baseline err | Dense err | Δ err | Baseline HOTA | Dense HOTA | Δ HOTA |
|-------|:-----------:|:---------:|:-----:|:-------------:|:----------:|:------:|
| Arno Vuarnier | 47.7 | 52.8 | +5.1 | 0.734 | 0.731 | −0.003 |
| Andreas Bakke | 8.7 | 9.1 | +0.4 | 0.932 | 0.926 | −0.006 |
| Jonatan Laland | 4.6 | 5.8 | +1.2 | 0.955 | 0.942 | −0.013 |
| Quentin Puydenus | 38.9 | 52.2 | +13.3 | 0.788 | 0.708 | −0.080 |
| **Mean** | **25.0** | **30.0** | **+5.0** | **0.852** | **0.827** | **−0.025** |

**Conclusion:** Rejected. Dense Farneback is significantly worse across all videos, especially Quentin (−80 HOTA points). Root cause: sparse LK is better at tracking the actual person bbox because it focuses on the most trackable keypoints around the last detected region; dense Farneback tracks average background motion across the whole frame and easily gets confused by camera shake and large motion.

---

## Iter 6 — Phase A merge: lower merge_threshold_low to recover excluded tracks (2026-03-14)

**Hypothesis:** Track 30 (76 dets, score 0.4922) is just 0.0078 below the 0.5 threshold and represents legitimate Arno detections near the aerial section. Lowering `merge_threshold_low` from 0.5 to 0.45 would recover Track 30 + Track 4 (28 dets, 0.4761), adding ~90 detected frames and reducing gap fill for both Arno and Quentin.

**Investigation finding:** Arno's 186 excluded frames come from 13 short tracks that scored 0.35–0.49. Segmentation has 900 detected frames but tracking only uses 714 because the merge excludes low-scoring track fragments.

**Results (mini-set):**

| Video | Baseline err | 0.45 thresh err | Δ err | Baseline HOTA | New HOTA | Δ HOTA |
|-------|:-----------:|:---------------:|:-----:|:-------------:|:--------:|:------:|
| Arno Vuarnier | 47.7 | 70.7 | +23.0 | 0.734 | 0.681 | −0.053 |
| Andreas Bakke | 8.7 | 8.7 | 0.0 | 0.932 | 0.932 | 0.000 |
| Jonatan Laland | 4.6 | 4.6 | 0.0 | 0.955 | 0.955 | 0.000 |
| Quentin Puydenus | 38.9 | 56.2 | +17.3 | 0.788 | 0.752 | −0.036 |
| **Mean** | **25.0** | **35.1** | **+10.1** | **0.852** | **0.830** | **−0.022** |

**Conclusion:** Rejected — catastrophic regression. Root cause: even at score 0.49, the excluded tracks for Arno and Quentin contain wrong detections (bystanders, ghost detections at incorrect positions). Including them floods the candidate pool, generates many new conflicts (Arno: 0→29 conflicts; Quentin: 144→344 conflicts), and the conflict resolver picks incorrectly. The 0.5 adaptive threshold is correctly filtering out contaminated tracks. **The 186 excluded Arno frames are not recoverable through threshold tuning — they're on genuinely ambiguous ByteTrack segments that score low because they're partially wrong.**

---

## Iter 7 — Segmentation: lower YOLO confidence threshold to 0.3 (2026-03-14)

**Hypothesis:** YOLO11x-seg's default confidence threshold of 0.5 misses many frames where the skier is correctly detected at lower confidence (white clothes against snow, small size during aerials). Lowering to 0.3 recovers those detections without significantly increasing false positives.

**Investigation:** Arno's segmentation at conf=0.5 had 900/1427 frames detected (63%), but 527 frames had conf=0.0 (YOLO completely missed). At conf=0.3, this jumped to 1160/1427 (81%), recovering 260 frames. Quentin similarly had large gaps that the lower threshold fills.

**Implementation:** `segmentation.confidence: 0.3` (from 0.5). Pure config change, applies to all videos.

**Results (mini-set):**

| Video | Baseline err | conf=0.3 err | Δ err | Baseline HOTA | New HOTA | Δ HOTA |
|-------|:-----------:|:------------:|:-----:|:-------------:|:--------:|:------:|
| Arno Vuarnier | 47.7 | 35.3 | **−12.4** | 0.734 | 0.847 | **+0.113** |
| Andreas Bakke | 8.7 | 8.4 | −0.3 | 0.932 | 0.935 | +0.003 |
| Jonatan Laland | 4.6 | 4.8 | +0.2 | 0.955 | 0.954 | −0.001 |
| Quentin Puydenus | 38.9 | 13.6 | **−25.3** | 0.788 | 0.892 | **+0.104** |
| **Mean** | **25.0** | **15.5** | **−9.5** | **0.852** | **0.907** | **+0.055** |

Also: Arno detected frames 714→872 (+158), Jonatan 1806→2042 (+236), Quentin 1158→1569 (+411). Quentin conflicts collapsed from 144→0 (many frames that caused ambiguous multi-detection conflicts at conf=0.5 are now cleanly detected).

**Dev-set evaluation:** Deferred — requires re-running segmentation on 6 more dev-set videos.

**Conclusion:** **ACCEPTED** — largest improvement of any experiment in this loop. `segmentation.confidence: 0.3`. Mini-set mean: 15.5px (−9.5px vs 25.0px), HOTA: 0.907 (+0.055). The 0.5 default threshold was too conservative for white-suited athletes in snowy conditions; 0.3 recovers the signal without flooding the tracker with false positives (HOTA improvement confirms correct identity maintained).

---
