# Experiment Backlog

Candidates not yet tried, or partially tried with clear remaining angles. Ordered roughly by expected ROI within each area. Remove items when tried; add new candidates as they surface.

---

## Tracking

### High priority — appearance / identity (known bottleneck)

**OSNet / FastReID appearance embeddings (Phase A soft scoring signal)**
- Rationale: Gabin Leonard and similar bystander-lock cases have a self-reinforcing feedback loop that only a trained appearance embedding can break. Simple color histograms (Iter 9) and size scoring (Iter 8) confirmed that hand-crafted appearance signals have no discriminative power in this domain. ReID is the natural next step.
- Expected effort: 3–5 days (model integration + fine-tuning on competition footage)
- Note: `w_color` infrastructure already wired in `tracker.py`; can be repurposed.

**Pose plausibility scoring (inline YOLO-Pose on Phase A candidates)**
- Rationale: Bystanders who are standing still or facing away will have clearly different pose keypoints from an active skier. Could break Gabin Leonard's conflict loop without requiring ReID training data.
- Expected effort: 2–3 days
- Note: More complex than initially proposed; requires per-candidate crop inference at conflict frames.

### High priority — aerial gap recovery (known bottleneck)

**SAMURAI for aerial-gap recovery (Phase B.1)**
- Rationale: SAM2-based zero-shot tracking is the highest-ceiling approach for the long aerial phases where LK/Farneback fail entirely. CoTracker3 infrastructure (`src/tracking/cotracker_fill.py`) exists and is a stepping stone.
- Expected effort: 5–7 days
- Note: CoTracker3 (Iter 4) failed because visibility scores drop below 0.5 during aerials. Try SAMURAI as the alternative for zero-shot aerial tracking. Also worth retrying CoTracker3 with a lower `min_visible` threshold before committing to SAMURAI.

**CoTracker3 with lower `min_visible` threshold**
- Rationale: Iter 4 rejected CoTracker3 with visibility threshold=0.5 (all points fell back to LK). A lower threshold (e.g. 0.2–0.3) may allow partial trajectory reconstruction during aerial occlusion.
- Expected effort: 0.5 days (config/param sweep, infrastructure already exists)
- Note: Fast to try before committing to the heavier SAMURAI approach.

### Medium priority — global optimization

**Global tracklet association (bidirectional tracking + Hungarian / min-cost flow)**
- Rationale: Offline bidirectional tracking + global optimization to merge tracklet fragments. Expected 10–15% error reduction on remaining identity switches. Higher computational cost (2× tracking passes).
- Expected effort: 5–7 days

### Medium priority — smoothing / interpolation

**Savitzky-Golay smoothing for gap interpolation**
- Rationale: Replace linear interpolation with polynomial smoothing for gap filling. Expected 5–10% error reduction on interpolated frames. Low computational cost.
- Expected effort: 0.5–1 day
- Note: Kalman smoothing experiments (Iter 7, Iter 10) confirmed the smoother is not the bottleneck for OF-derived positions — this candidate is most relevant for short gaps where linear interpolation currently runs, not long aerial phases.

### Low priority — already explored, diminishing returns

**OC-SORT observation-centric Kalman re-init after long gaps**
- Rationale: Listed as Experiment C in loop-20260313 candidate list. Iter 7 tested a closely related approach (RTS guard at gap boundaries) and found it null. A true OC-SORT re-init (reinitialising the full covariance matrix from the post-gap detection) is slightly different and not yet tried exactly.
- Expected effort: 1 day
- Note: Iter 7 and Iter 10 together confirm the Kalman is not the bottleneck; deprioritise unless OF gap-fill quality improves first.

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
