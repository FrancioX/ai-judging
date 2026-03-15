# Experiment Backlog

Candidates not yet tried, or partially tried with clear remaining angles. Ordered roughly by expected ROI within each area. Remove items when tried; add new candidates as they surface.

---

## Venue Mapping

**Current best (valid)**: Background OF + PCHIP, one fixed start-gate anchor per venue.
Knözinger: 28.5px LOO mean, 92% within 50px. Script: `scripts/venue_camflow.py`.

**Annotate more videos for venue mapping (dev set)**
- Rationale: Only 2 videos have venue GT (Andreas Bakke, Knözinger). Need broader validation across different slope types and zoom levels to confirm 28-53px LOO is representative.
- Effort: 15 annotations × N videos via `scripts/annotate_venue.py` (~3 min/video)

**Optimal annotation placement vs. uniform spacing**
- Rationale: Sparsity test used evenly-spaced annotations. Placing 5 annotations at path inflection points (where the athlete turns or accelerates) could improve LOO accuracy vs. uniform placement with the same budget.
- Effort: 0.5 day

**Extend tracking to full video run (>990 frames)**
- Rationale: Tracking currently stops at frame 990 for one evaluation video while GT extends to 1370. Tracking-guided interpolation could help for frames 990-1370 if full tracking is available.
- Effort: Investigate why tracking stops early; may be a pipeline cutoff

**Slope silhouette / sky-horizon matching (automatic anchor)**
- Rationale: The sky-slope boundary is a distinctive 1D profile visible from both broadcast and venue viewpoints. When sky is visible in the frame, matching this silhouette against the venue image could replace the manual start-gate click. Not tried.
- Expected effort: 1-2 days
- Note: Requires sky to be consistently visible in early-run frames. Standard NCC/SIFT fail (Exp 3) because the broadcast camera and venue image are from different physical positions — but silhouette matching is a 1D geometric feature that is more viewpoint-robust.

**Venue reference image from broadcast camera (eliminates viewpoint problem)**
- Rationale: If a single wide-angle image is captured with the broadcast camera (zoomed fully out) at competition start, it shares the exact viewpoint with the video. Template matching would then work precisely without any manual annotation.
- Effort: Operational (one image per venue per day); no code changes needed beyond adding the wide-angle image as `venue_image.png`.

**Wire venue_camflow into main pipeline as stage 7**
- Rationale: `scripts/venue_camflow.py` is standalone; should be wired into `src/pipeline.py` as a proper stage writing `output/venue_mapping/<stem>/` and a JSON manifest.
- Effort: 2-3 hours

---

## Tracking

### High priority — segmentation (easy wins remaining)

**`imgsz: 1920` for YOLO11x-seg (requires CUDA hardware)**
- Rationale: Loop14 Iter 11 confirmed imgsz=1920 fails on MPS (`Output channels > 65536`). On CUDA, this should recover Arno's 293-frame invisible gap and similar cases. Arno: 35.3px → potentially 15–25px.
- Expected effort: 0.5 days (config change only; requires GPU node)
- Note: Arno has 148 frames with conf=0.0 even at 1920px, so ceiling is not zero. But 145 frames were detecting a different person; full resolution might recover some of those.

**Full 18-video evaluation with current best config**
- Rationale: Loop14 achieved 13.0px / 0.924 HOTA on 10-video dev-set. Need to verify on all 18 videos to establish a reliable aggregate and identify any remaining failure cases not in the dev-set.
- Expected effort: 0.5 days (run frames+segmentation+tracking for 8 remaining videos)

---

### High priority — appearance / identity (known bottleneck)

**OSNet / FastReID appearance embeddings (Phase A soft scoring signal)**
- Rationale: Gabin Leonard and similar bystander-lock cases have a self-reinforcing feedback loop that only a trained appearance embedding can break. Simple color histograms (Iter 9) and size scoring (Iter 8) confirmed that hand-crafted appearance signals have no discriminative power in this domain. ReID is the natural next step.
- Expected effort: 3–5 days (model integration + fine-tuning on competition footage)
- Note: `w_color` infrastructure already wired in `tracker.py`; can be repurposed.

**`of_synthetic_confidence: 0.0` for bystander-lock cases**
- Rationale: Loop14 Iter 15 found null on mini-set (no conflicts in aerial gaps). But for Gabin Leonard's bystander lock, synthetic OF candidates may reinforce the wrong track. Worth testing on the full dev-set specifically for Gabin.
- Expected effort: 0.5 days (config change, run Gabin tracking + evaluate)

**Pose plausibility scoring (inline YOLO-Pose on Phase A candidates)**
- Rationale: Bystanders who are standing still or facing away will have clearly different pose keypoints from an active skier. Could break Gabin Leonard's conflict loop without requiring ReID training data. Gabin's 52.2px is the main remaining outlier.
- Expected effort: 2–3 days
- Note: More complex than initially proposed; requires per-candidate crop inference at conflict frames.

### High priority — aerial gap recovery (known bottleneck)

**SAMURAI for aerial-gap recovery (Phase B.1)**
- Rationale: SAM2-based zero-shot tracking is the highest-ceiling approach for the long aerial phases where LK/Farneback fail entirely. CoTracker3 infrastructure (`src/tracking/cotracker_fill.py`) exists and is a stepping stone.
- Expected effort: 5–7 days
- Note: CoTracker3 (Iter 4) failed because visibility scores drop below 0.5 during aerials. Try SAMURAI as the alternative for zero-shot aerial tracking. Also worth retrying CoTracker3 with a lower `min_visible` threshold before committing to SAMURAI.

**CoTracker3 with lower `min_visible` threshold** ~~DONE — Null~~
- Loop14 Iter 10 confirmed: CoTracker3 returns binary {0,1} visibility (internally thresholded). Lowering min_visible_score has no effect. Athletes are genuinely invisible during aerials — CoTracker cannot track what is not visible.

### Medium priority — global optimization

**Global tracklet association (bidirectional tracking + Hungarian / min-cost flow)**
- Rationale: Offline bidirectional tracking + global optimization to merge tracklet fragments. Expected 10–15% error reduction on remaining identity switches. Higher computational cost (2× tracking passes).
- Expected effort: 5–7 days

### Medium priority — smoothing / interpolation

**Savitzky-Golay smoothing for gap interpolation** ~~DEPRIORITIZED~~
- Loop14 confirmed: mini-set videos have no short internal gaps (<10 frames). Bidirectional OF blending outperforms linear interpolation even for long aerial gaps (Iter 13: +12.6px for linear vs OF). SG would only help sub-10-frame gaps; these are negligible for the current corpus. Remove from active backlog unless short-gap videos emerge.

### Low priority — already explored, diminishing returns

**OC-SORT observation-centric Kalman re-init after long gaps** ~~DONE — Null~~
- Loop14 Iter 14 (`kalman_reinit_gap: 50`) confirmed completely null. High `r_interp_pos: 40.0` already prevents gap-fill corruption. Remove from backlog.

---

## Pose Estimation (2D → 3D Lifting)

*No experiments run yet. Tracking is the current focus.*

**MotionBERT full integration (replace stub)**
- Rationale: The 3D lifting stage currently writes a stub. Completing it is a prerequisite for any pose-quality evaluation.
- Expected effort: 2–3 days

**Multi-view consistency (if second camera angle available)**
- Rationale: Freeride competitions are sometimes broadcast with multiple camera angles. If a second angle is available, triangulation could replace monocular lifting entirely.
- Expected effort: Unknown; depends on data availability.

---

## Segmentation

*No experiments run yet.*

**Fine-tune YOLOv11x-Seg on ski/snowboard footage**
- Rationale: Off-the-shelf segmentation may confuse skiers with equipment or background. Domain fine-tuning could reduce false positives and improve bounding box quality upstream of tracking.
- Expected effort: 3–5 days (requires labelled segmentation data)
