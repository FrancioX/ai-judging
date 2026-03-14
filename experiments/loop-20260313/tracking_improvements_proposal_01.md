# Tracking Improvements Proposal 01

**Date:** 2026-03-13
**Current best:** 23.5 px mean error, 0.869 HOTA (Exp 20 + Exp 21a, 18 videos)
**Primary failure case:** Arno Vuarnier ~48 px — aerial-phase identity switches with nearby bystander

---

## Context: What Has Already Been Tried

The following approaches were tested across Experiments 1–21 and either reverted or found neutral:

| Approach | Outcome |
|---|---|
| Velocity consistency scoring | Redundant with OF agreement signal; kept at w=0.4 only for manifest data |
| Camera motion compensation (ORB/ECC homography) | Cannot self-bootstrap — velocity history gets poisoned before CMC can help |
| Previous-detection-only identity guard | Strong regression in no-OF mode |
| Confidence-gated merge candidates | Too aggressive — removes valid low-conf skier detections |
| High-confidence OF re-anchoring (min_conf=0.7) | Neutral with `of_trace_filter_enabled=false` — mechanism no longer needed |
| Full OF in Phase B for short gaps | Latches onto background motion; conditional gap-fill (≥10 frames) is better |

What works and is in production (Exp 20/21a config):
- OF trace as soft Phase A score signal (filter disabled)
- Adaptive merge threshold (0.5 low / 0.6 high, based on detection rate)
- Conditional gap-fill (OF only for gaps ≥10 frames, drift guard 200px)
- Kalman + RTS smoother

---

## Literature Findings

### SkiTB 2025 (WACV 2025) — Direct Benchmark

**"Tracking Skiers from the Top to the Bottom"** (Dunnhofer et al., WACV 2024 / CVIU 2024) introduced the SkiTB dataset: 300 videos, 353K frames, three disciplines (alpine, ski jumping, freestyle). Key finding: freestyle is the hardest (aerial phases, fastest motion). Most generic trackers fail at camera switches and during jumps; the best performers use re-initialization.

**SkiTB Visual Tracking Challenge 2025** winning entry: **ReID-SAM** (arXiv 2503.01907), F1 = 0.870.
- Combines SAMURAI (SAM2 + Kalman memory scoring) + OSNet ReID.
- For each camera segment: extract a ReID feature vector for the tracked result; if cosine similarity to the expected skier drops below threshold, trigger a YOLO11 scan, re-prompt SAMURAI with the best-similarity detection, run forward+backward tracking.
- This is essentially the same failure mode as Arno Vuarnier: aerial phase → identity loss → bystander confusion on re-entry.

---

## Proposed Experiments (Prioritized)

### Experiment A — Pose Plausibility Bonus in Phase A Scoring

**Effort:** < 1 day | **New deps:** None | **Expected gain:** Small but free

**What:** Add a small bonus (0.1–0.2) to Phase A conflict-resolution candidates that have ≥8 visible YOLO-Pose keypoints in a non-upright posture (hip center clearly below shoulder center, or any dynamic crouching configuration).

**Why:** The pipeline already runs YOLO11x-pose on every frame. A standing bystander vs. a skiing athlete in a tuck/crouch is trivially distinguishable from skeleton geometry. This signal is sitting in the output JSON and currently unused in Phase A.

**Implementation sketch:**
- Load the existing YOLO-pose manifest for the current video.
- For each conflict frame, look up detected keypoints for each candidate bbox.
- Score: `pose_bonus = 0.15` if candidate has ≥8 confident keypoints AND `hip_y > shoulder_y + threshold`.
- Add as `w_pose × pose_bonus` to the Phase A candidate scorer.

**Risk:** Near zero. It only fires on conflict frames where ≥2 candidates exist, and the bonus magnitude (0.1–0.2) is small relative to the 7-axis combined score.

---

### Experiment B — OSNet ReID as Phase A Scoring Signal

**Effort:** 1–2 days | **New deps:** `torchreid` (or direct OSNet inference) | **Expected gain:** High

**What:** Maintain a rolling reference appearance embedding (512-d, EMA over last 30 high-confidence detections, conf > 0.5). Add cosine similarity to this reference as a 7th axis in the Phase A conflict scorer with weight ~0.3.

**Why:** The SkiTB 2025 winner used exactly this approach. The skier wears a distinctive helmet and suit; the bystander doesn't. Cosine similarity cleanly separates them in appearance space. OSNet-x0.25 is 0.6M parameters and runs hundreds of fps even on CPU — negligible overhead.

**Implementation sketch:**
```python
import torchreid
model = torchreid.models.build_model('osnet_x0_25', num_classes=1000, pretrained=True)
# Extract 256×128 crop for each candidate detection
# Normalize: ImageNet mean/std
# embedding = model(crop)  # → (512,)
# score = cosine_similarity(embedding, reference_embedding)
```

Reference embedding update: exponential moving average — `ref = alpha * ref + (1-alpha) * new_emb` where `alpha=0.95`, triggered only when detection confidence > 0.5.

**Risk:** Medium. OSNet was trained on pedestrian ReID datasets (person re-identification), not skiers specifically. Performance in snow/outdoor conditions needs validation. Start with a low weight (0.2) and sweep upward.

**Repo:** [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid), model: `osnet_x0_25_imagenet`.

---

### Experiment C — OC-SORT Observation-Centric Kalman Re-initialization

**Effort:** 2–3 days | **New deps:** None | **Expected gain:** Medium

**What:** After Phase B gap-fill, instead of propagating the Kalman state through the gap (which accumulates velocity error), re-anchor from the two gap endpoints: the last pre-gap detection and the first post-gap detection. This is the core insight of OC-SORT (CVPR 2023).

**Why:** During Arno's aerial phase, the Kalman filter extrapolates for ~60 frames. The state's predicted position at re-entry may be far from the actual landing zone, causing the re-association to fail or match the wrong person. The RTS backward smoother mitigates this partially, but OC-SORT's re-initialization is a different mechanism: it resets the state from observed endpoints rather than smoothing through the gap.

**Implementation sketch:**
- After Phase B fills a gap of length ≥ N frames (e.g., 20), detect the two anchor detections.
- Re-initialize Kalman state: set position from anchor, compute velocity from gap-fill trajectory endpoints, reset acceleration to zero.
- Apply Phase C RTS smoother from the reset state.

**Reference:** Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking," CVPR 2023.

---

### Experiment D — SAMURAI for Aerial-Gap Recovery

**Effort:** 1–2 weeks | **New deps:** SAM2 / SAMURAI | **Expected gain:** Highest ceiling

**What:** Run SAMURAI in a targeted recovery mode, triggered only when the tracker detects a contiguous detection gap ≥20 frames. Initialize SAMURAI from the last confident detection before the gap (bbox prompt). Let it propagate the skier's visual object (suit texture + shape) through the gap using its internal memory bank. Use the SAMURAI output trajectory as Phase B gap-fill instead of Lucas-Kanade for these long gaps.

**Why:** This is the most direct fix for the aerial-phase failure mode. During a gap where YOLO produces no detections, Lucas-Kanade can drift to background features or nearby bystanders. SAM2 tracks the *visual object* (skier's distinctive outfit) without needing YOLO detections — it propagates the mask from its memory bank of past frames.

**MPS caveat:** SAM2's MPS support is partial. Community reports ~5–8 fps on M2/M3 for video mode. This is workable for offline batch processing (not real-time), but `sam2-hiera-small` should be tested first. The targeted mode (only on long-gap windows) limits the frames SAM2 actually processes.

**Integration approach:**
1. Detect all gaps ≥20 frames in Phase B.
2. For each gap, slice the raw frames window (last_detection - 5 frames to first_post_gap_detection + 5 frames).
3. Run SAMURAI initialized from the pre-gap bbox prompt.
4. Extract the center trajectory from the SAMURAI mask outputs.
5. Use as gap-fill trajectory; validate with ReID (Experiment B) before accepting.

**Repos:**
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- [yangchris11/samurai](https://github.com/yangchris11/samurai)

**Reference:** "SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory," arXiv 2411.11922 (Nov 2024). Achieved +7.1% AUC on LaSOT-ext vs vanilla SAM2 with no retraining.

---

### Experiment E — CoTracker3 Gap Fill (Phase B Enhancement)

**Effort:** 3–5 days | **New deps:** `co-tracker` (pip) | **Expected gain:** Medium

**What:** Replace Phase B Lucas-Kanade gap fill with CoTracker3's offline mode for gaps ≥10 frames. Track 5–8 body keypoint locations (initialized at the last confident detection before the gap) through the gap; use the weighted average of surviving keypoints as the body-center trajectory.

**Why CoTracker3 > Lucas-Kanade:** CoTracker3 uses a transformer that jointly tracks all points with cross-track attention — point trajectories constrain each other, making it more robust to rotation during aerials and partial occlusion. LK's local gradient tracking degrades rapidly when the skier rotates or when the background texture is uniform (snow). CoTracker3's offline mode can handle long occlusions because it attends to the full temporal context, not just adjacent frames.

**Implementation sketch:**
```python
from cotracker.predictor import CoTrackerPredictor
model = CoTrackerPredictor(checkpoint=None)  # auto-downloads
# queries: (1, N_points, 3) — [t, x, y] at last pre-gap frame
pred_tracks, pred_visibility = model(video_frames, queries=queries)
# pred_tracks: (1, T, N_points, 2) — trajectory per point
# Center = mean of visible points (pred_visibility > 0.5)
```

**Risk:** CoTracker3 is trained on synthetic + pseudo-labeled real video; performance on fast-motion snow scenes needs validation. Also: initializing from body keypoints requires the pose output for the pre-gap frame, which requires pipeline ordering changes (pose before tracking, or a two-pass approach for gap frames only).

**Repo:** [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker), pip: `pip install cotracker`.

---

## Recommended Execution Order

```
A (pose bonus) → B (OSNet ReID) → evaluate 18 videos
  → if Arno still > 30px: C (OC-SORT re-init) → evaluate
  → if Arno still > 30px: D (SAMURAI) → evaluate
E (CoTracker3) can be explored in parallel with D
```

Experiments A and B together address the specific Arno failure mode (post-aerial re-identification) with the lowest implementation cost. D is the ceiling-raiser if A+B are insufficient, but requires MPS validation.

---

## What to Skip (for Now)

| Approach | Reason |
|---|---|
| TrackFormer / MOTR / MOTRv2 | Full pipeline replacement; not practical on MPS; pedestrian-domain only |
| BoT-SORT / StrongSORT as drop-in | Better to extract their sub-components (ReID, OC-SORT re-init) individually |
| GTA global tracklet association | Designed for 22-player team sports; over-engineered for single-target tracking |
| Depth Anything V2 depth signal | Unreliable in textureless snow scenes; ViT depth on flat white surfaces is noisy |
| Camera motion compensation (full) | Tried in Exp 3 — cannot self-bootstrap; CMC idea itself is sound but the velocity-consistency integration failed |

---

## References

- Dunnhofer et al., "Tracking Skiers from the Top to the Bottom," WACV 2024 / CVIU 2024.
- SkiTB Visual Tracking Challenge 2025, WACV 2025 workshop — [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/20897).
- "Technical Report for ReID-SAM on SkiTB Visual Tracking Challenge 2025," arXiv 2503.01907.
- Yang et al., "SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory," arXiv 2411.11922 (Nov 2024).
- Ravi et al., "SAM 2: Segment Anything in Images and Videos," arXiv 2408.00714 (Aug 2024).
- Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking," CVPR 2023.
- Karaev et al., "CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos," arXiv 2410.11831 (Oct 2024).
- Zhou & Recovered, "Omni-Scale Feature Learning for Person Re-Identification" (OSNet), ICCV 2019. Impl: [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).
