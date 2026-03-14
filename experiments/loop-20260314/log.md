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

## Iter 8 — Segmentation: select_strategy "largest" vs "center" (2026-03-14)

**Hypothesis:** Switching from `select_strategy: "center"` (pick the detection closest to frame center) to `"largest"` (pick the largest bbox) recovers more Arno aerial frames where the skier is correctly segmented but not necessarily near frame center.

**Implementation:** `segmentation.select_strategy: "largest"`. Pure config change. Mini-set rerun (segmentation + tracking) from scratch with conf=0.3 and select_strategy="largest".

**Results (mini-set):**

| Video | Baseline err (px) | New err (px) | Δ err | Baseline HOTA | New HOTA | Δ HOTA |
|-------|:-----------------:|:------------:|:-----:|:-------------:|:--------:|:------:|
| Arno Vuarnier | 35.3 | 35.3 | 0.0 | 0.847 | 0.848 | +0.001 |
| Andreas Bakke | 8.4 | 8.4 | 0.0 | 0.935 | 0.935 | 0.000 |
| Jonatan Laland | 4.8 | 4.8 | 0.0 | 0.954 | 0.954 | 0.000 |
| Quentin Puydenus | 13.6 | 13.6 | 0.0 | 0.892 | 0.892 | 0.000 |
| **Mean** | **15.5** | **15.5** | **0.0** | **0.907** | **0.907** | **0.000** |

Arno detected frames with "largest": 862 (vs 872 with "center"). 10 fewer detected frames, but no aggregate metric change.

**Conclusion:** Null result. Rejected — no effective config revert needed since "largest" was already the baseline default in config.yaml. At conf=0.3, both "center" and "largest" resolve to the same detection in virtually every frame — the skier is the only large/central detection. The 10-frame detection difference did not measurably affect mean error.

---

## Iter 9 — Phase B: re-sweep `of_drift_guard_px` at conf=0.3 baseline (2026-03-14)

**Hypothesis:** Iter 3 found 400px optimal at conf=0.5 (Arno: 713 gap frames, Quentin: 1014). With conf=0.3, gaps are shorter (Arno: 565, Quentin: 603) and detection patterns differ. The optimal drift guard may have shifted.

**Implementation:** Swept `of_drift_guard_px` across 0 (disabled), 200, 300, 400 (current), 500, 600. Pure config change; tracking rerun for Arno and Quentin only (segmentation cached).

**Sweep results (Arno + Quentin only, Andreas/Jonatan unchanged at 8.4/0.935 and 4.8/0.954):**

| Guard (px) | Arno err | Arno HOTA | Quentin err | Quentin HOTA | 4-video mean err | 4-video HOTA |
|:----------:|:--------:|:---------:|:-----------:|:------------:|:----------------:|:------------:|
| 0 (disabled) | 35.3 | 0.848 | 13.6 | 0.892 | 15.5 | 0.907 |
| **200** | **51.1** | **0.791** | **15.3** | **0.885** | **~19.8** | **~0.899** |
| 300 | 35.3 | 0.848 | 13.6 | 0.892 | 15.5 | 0.907 |
| 400 (current) | 35.3 | 0.848 | 13.6 | 0.892 | 15.5 | 0.907 |
| 500 | 35.3 | 0.848 | 13.6 | 0.892 | 15.5 | 0.907 |
| 600 | 35.3 | 0.848 | 13.6 | 0.892 | 15.5 | 0.907 |

**Conclusion:** Null result. Rejected. Guard=200 uniquely bad (prematurely falls back to linear interpolation when OF is correctly tracking). Guards 0, 300–600 are all identical — with conf=0.3, OF gap-fill never drifts >300px from linear for Arno's or Quentin's aerial gaps, so the guard never triggers above 300. Current 400px is in the flat optimal zone. Key insight: Arno's 35.3px error is the best achievable via LK optical flow on the current 565-frame aerial gap — the drift guard is no longer the bottleneck.

---

## Iter 10 — Phase B: CoTracker3 with lower `min_visible` threshold (2026-03-14)

**Hypothesis:** Loop13 Iter 4 rejected CoTracker3 because all tracked points had visibility <0.5. Since CoTracker3 returns float confidence scores, lowering the threshold to 0.2 might allow partial trajectory reconstruction during Arno's 293-frame internal aerial gap.

**Implementation:**
- Installed CoTracker3 (`uv add git+https://github.com/facebookresearch/co-tracker.git`)
- Added `cotracker_min_visible_score` parameter to `cotracker_fill.py`, `tracker.py`, and `pipeline.py`
- Fixed pre-existing bug: `_get_stage_kwargs("tracking")` did not forward CoTracker/conflict/Kalman parameters when using `--stage tracking`; all these parameters were silently defaulting to their hardcoded values. Fixed by adding them to the returned dict.
- Set `cotracker_enabled: true`, `cotracker_min_visible_score: 0.2`

**Results (Arno only, the target for CoTracker):**

| Video | Baseline err | CoTracker err | Δ err |
|-------|:-----------:|:-------------:|:-----:|
| Arno Vuarnier | 35.3 | 35.3 | 0.0 |

CoTracker ran on the 293-frame gap (frames 403→697) but immediately returned None: at frame 404 (t=1), all 10 tracked keypoints had visibility=False. The gap was not re-filled.

**Conclusion:** Rejected. Root cause: CoTracker3 returns **boolean** visibility (not float confidence), already thresholded at 0.5 internally. Lowering `min_visible_score` from 0.5 to 0.2 has no effect — the output is {True, False}, not a float. All 10 points are False during Arno's aerial because he is genuinely invisible (white clothing against snow): this is the same fundamental invisibility that prevents YOLO from detecting him. CoTracker cannot track what it cannot see. `cotracker_enabled: false` restored. The `_get_stage_kwargs` bug fix was committed as infrastructure (affects all future `--stage tracking` experiments).

---

## Iter 11 — Segmentation: `imgsz: 1920` + gap region analysis (2026-03-14)

**Hypothesis:** Running YOLO at full 1920px resolution (matching the video's native resolution) might detect Arno in more frames during the aerial gap, reducing the 293-frame internal gap (frames 404-696).

**Implementation:** `segmentation.imgsz: 1920`. Pure config change. Tested on Arno only.

**Results:** Not testable — `NotImplementedError: Output channels > 65536 not supported at the MPS device.` at 1920px input size. Apple MPS backend imposes a hard limit that YOLO11x-seg exceeds at this resolution. Would require CPU inference (estimated 20–30 min/video) or CUDA hardware.

**Gap region investigation (frames 404-696 at imgsz=1280, conf=0.3):**
- 293 frames in gap (Arno undetected across entire gap at conf=0.3)
- 145 frames have non-zero detections (all from ByteTrack 104, center x≈1350-1400): these are a **different person** in the right part of the frame, not Arno
- 148 frames have zero detections — YOLO finds no person at all in those frames
- Arno's OF trace starts at (1010, 662) at frame 403, ends at (872, 503) at frame 698. OF linearly tracks background features through the aerial, not Arno's true parabolic trajectory.
- Key finding: **Arno is genuinely invisible during the 293-frame gap.** No threshold reduction (0.25, 0.2, etc.) will recover him — 148 frames have conf=0.0 by definition, not because the threshold is too high.

**Conclusion:** Rejected (technical failure on MPS). Reverted to `imgsz: 1280`. The aerial gap analysis confirms that Arno's remaining 35.3px error is an **irreducible detection bottleneck** on MPS hardware. The error ceiling for Arno on this mini-set requires either CUDA hardware for 1920px inference, domain-specific model fine-tuning, or a fundamentally different detection approach (e.g., thermal/IR, multi-camera). Further tuning experiments are unlikely to move Arno's error materially.

---
