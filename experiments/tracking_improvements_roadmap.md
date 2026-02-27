# Tracking Improvements Roadmap

## Current State (Baseline)

**Evaluation Results (26 Feb 2026)**:
- Arno Vuarnier: 188.4px mean error (59.8% within 50px)
- Andreas Bakke: 93.4px mean error (77.7% within 50px)
- Lach Powell: 40.4px mean error (92.8% within 50px)
- **Overall**: 107.4px mean error, 100% detection rate

**Primary Problem**: Identity confusion when multiple people appear in frame during aggressive camera panning/zooming.

---

## Proposed Improvements (Ordered by Implementation Priority)

### **1. HOTA Metric Addition** ⚡ [1-2 days]

**Computational Cost**: Negligible (evaluation only)
**Expected Impact**: Better insight into identity preservation

**What**: Add Higher Order Tracking Accuracy (HOTA) metric alongside current pixel-error metrics.

**Why**:
- Current metrics (mean_error_px, P90, P95) measure spatial accuracy but not identity preservation quality
- HOTA balances detection quality (DetA) and association quality (AssA)
- Better suited for judging applications where maintaining correct athlete identity over entire run is critical

**Implementation**:
- New module: `src/tracking/hota.py`
- Extend `evaluate.py` to compute: HOTA, DetA, AssA
- Report in batch evaluation summary

**Technical Details**:
- Center-distance based matching (threshold ~50px to match bbox IoU ≈ 0.5)
- Sweep distance thresholds [10, 20, 30, ..., 100px]
- HOTA = √(DetA × AssA) averaged across thresholds
- Pure NumPy implementation (no external deps)

---

### **2. Camera Motion Compensation (CMC)** ⚡⚡⚡ [3-5 days]

**Computational Cost**: +10-20ms/frame (ORB features on background)
**Expected Impact**: 30-40% error reduction on videos with aggressive camera motion (e.g., Arno)

**What**: Decompose observed motion into camera motion (pan/tilt/zoom) + object motion (skier).

**Why**:
- Root cause of spatial jumps: Camera pans to follow skier → bounding boxes "teleport" across frame → ByteTrack loses track
- Current identity guard uses absolute frame coordinates → threshold violations on camera motion
- CMC transforms all detections into stabilized coordinate system → smoother tracks, better identity preservation

**Implementation Strategy**:
- Compute per-frame global homography/affine transform using background features
- Subtract camera motion from optical flow measurements
- Apply compensation in `_fill_gaps_optical_flow()` before bbox propagation

**Methods** (fastest to slowest):
1. **ORB/AKAZE features** (Recommended) - Fast, rotation-invariant, ~10ms/frame
2. **ECC (Enhanced Correlation Coefficient)** - Medium speed, robust to illumination
3. **Dense Farneback flow** - Slower, but already computed for gaps

**New Function**:
```python
def _compute_camera_motion(
    img_prev: np.ndarray,
    img_next: np.ndarray,
    mask_roi: np.ndarray | None = None,  # exclude ROI (skier) from matching
    method: str = "orb",  # orb | akaze | ecc | farneback
) -> tuple[float, float, float, str]:
    """Estimate global camera translation and rotation.

    Returns: (dx, dy, d_theta, method_used)
    """
```

**Integration Points**:
- **Primary**: Inject in `_fill_gaps_optical_flow()` (line 834) before gap filling loop
- Store camera motion per frame: `camera_motion: dict[int, tuple[float, float, float]]`
- In `_compute_flow_displacement()`, subtract camera motion from measured flow
- Update `flow_displacement` field in tracking.json to reflect compensated motion

**Configuration** (add to `config.yaml`):
```yaml
tracking:
  camera_motion_compensation:
    enabled: true
    method: "orb"  # orb | akaze | ecc | farneback
    exclude_roi_margin: 1.5  # exclude 1.5× bbox area from feature matching
    min_features: 20  # minimum matched features (fallback to none if lower)
    ransac_threshold: 3.0  # outlier rejection threshold for homography
```

---

### **3. Kalman Filter Tuning** ⚡ [1 day]

**Computational Cost**: Negligible (change motion model only)
**Expected Impact**: 5-10% error reduction

**What**: Replace constant velocity model with constant acceleration model in velocity-aware smoothing.

**Why**:
- Skiing involves gravity, terrain changes, jumps → non-constant velocity
- Current Kalman-like filter in `_smooth_bboxes_velocity_aware()` assumes constant velocity
- Constant acceleration model better captures parabolic trajectories (jumps, drops)

**Implementation**:
- Extend state vector from [x, y, vx, vy] to [x, y, vx, vy, ax, ay]
- Update prediction step to include acceleration term
- Use OF velocities to estimate acceleration via finite differences
- Minimal code change (~20-30 lines in existing function)

**Optional Enhancement**: Add physics constraints
- Max acceleration from skiing biomechanics (~2g lateral, ~1g vertical)
- Clip unrealistic accelerations

---

### **4. Trajectory Smoothing (Savitzky-Golay)** ⚡ [1 day]

**Computational Cost**: Negligible post-processing
**Expected Impact**: 5-10% error reduction on interpolated frames

**What**: Replace linear interpolation with polynomial smoothing for gap filling.

**Why**:
- Linear interpolation assumes constant velocity → errors on curved trajectories
- Savitzky-Golay fits local polynomial → respects physics (smooth acceleration)
- Better than GPR (overkill) or splines (overfitting risk)

**Implementation**:
- Apply to center coordinates after gap filling
- Window size: 7-11 frames (config parameter)
- Polynomial order: 2 (quadratic) or 3 (cubic)
- Use `scipy.signal.savgol_filter` if available, else implement manually

**Configuration**:
```yaml
tracking:
  smooth_interpolated: true
  savgol_window: 9  # must be odd
  savgol_order: 2   # polynomial degree
```

---

### **5. Global Tracklet Association** ⚡⚡⚡⚡ [5-7 days]

**Computational Cost**: 2× tracking passes + graph optimization
**Expected Impact**: 10-15% error reduction on remaining identity switches

**What**: Offline bidirectional tracking + global optimization to link tracklet fragments.

**Why**:
- After CMC fixes most switches, some fragmentation remains (occlusions, missed detections)
- Forward-only tracking is greedy → suboptimal over full video
- Offline advantage: can "look into future" to fix past mistakes

**Implementation Strategy**:
1. **Forward pass**: Current tracking pipeline → tracklets_forward
2. **Backward pass**: Reverse frame order, run tracking → tracklets_backward
3. **Global linking**: Hungarian algorithm or Min-Cost Flow to merge fragments
   - Cost = spatial gap + pose similarity (if poses available) + temporal discontinuity
4. **Output**: Single unified track

**Methods**:
- **Hungarian Assignment** - Fast, O(n³), good for <100 tracklets
- **Min-Cost Flow** - Scales better, O(n² log n)
- **GTA-Link** - State-of-art, graph transformer (complex, maybe overkill)

**New Functions**:
```python
def track_bidirectional(...) -> tuple[list[Tracklet], list[Tracklet]]:
    """Run tracking forward and backward."""

def link_tracklets(
    forward: list[Tracklet],
    backward: list[Tracklet],
    cost_spatial_weight: float = 0.5,
    cost_temporal_weight: float = 0.3,
    cost_pose_weight: float = 0.2,
) -> list[Tracklet]:
    """Global optimization to merge tracklet fragments."""
```

---

### **6. Pose-Guided Re-Identification** ⚡⚡⚡⚡⚡ [7-10 days + architectural redesign]

**Computational Cost**: 3-10× pose computation OR 2-pass system
**Expected Impact**: 20-30% error reduction, but requires major changes

**What**: Use skeletal pose similarity to distinguish between people when spatial methods fail.

**Why**:
- Two skiers might have similar appearance/location
- Their skeletal proportions and pose "signature" are unique
- Helps resolve ambiguous cases where CMC + spatial tracking aren't enough

**The Problem** (pipeline order):
```
Current: Segmentation → Tracking (select best) → Pose (on selected track only)
Required: Segmentation → Pose (on ALL people!) → Tracking (use poses)
```

**Cost Analysis**:
- Current: ~1 pose estimation per frame
- Required: ~3-10 pose estimations per frame (all detected persons)
- **3-10× increase in pose computation time**
- YOLO-Pose is already one of the heavier operations

**Implementation Options**:

**Option A: Compute all poses upfront** [Expensive]
- Run pose estimation on every detection in segmentation manifest
- Store poses per (frame_id, track_id)
- Use poses in track scoring and identity guard
- **Cost**: 3-10× pose time, but cleanest architecture

**Option B: Two-pass tracking** [Complex]
- First pass: Current tracking (spatial only)
- Identify ambiguous frames (multiple high-scoring candidates)
- Second pass: Compute poses only for ambiguous cases, re-track
- **Cost**: 2× tracking overhead + selective pose computation

**Pose Similarity Metrics**:
- **PCK (Percentage of Correct Keypoints)**: Match joint locations within threshold
- **OKS (Object Keypoint Similarity)**: Weighted by joint importance + bbox area
- **Limb Ratio Signature**: [torso_length, leg_length, arm_length] ratios (scale-invariant)

**Recommended Approach**:
- Implement **only after** CMC + Kalman + Global Association
- Start with Option B (two-pass) to minimize overhead
- Profile to ensure pose computation is acceptable (target: <2× total pipeline time)

---

## Implementation Timeline

### **Phase 1: Evaluation Improvements** [Week 1]
- Day 1-2: HOTA metric implementation
- Establish better baseline for measuring identity preservation

### **Phase 2: Core Tracking Fixes** [Week 2-3]
- Day 3-7: Camera Motion Compensation (CMC)
- Day 8: Kalman filter tuning
- Day 9: Savitzky-Golay smoothing
- **Checkpoint**: Re-evaluate all 3 videos, expect ~40-50% error reduction

### **Phase 3: Advanced Optimization** [Week 4-5]
- Day 10-16: Global tracklet association
- **Checkpoint**: Re-evaluate, expect incremental 10-15% improvement

### **Phase 4: Pose Re-ID (Optional)** [Week 6-8]
- Only if Phase 2-3 results are insufficient
- Requires architectural discussion and profiling

---

## Success Criteria

### **Minimum Viable**:
- Mean error <70px across all videos (35% improvement)
- >85% within 50px threshold (up from 76%)
- HOTA score >0.75

### **Target**:
- Mean error <50px across all videos (53% improvement)
- >90% within 50px threshold
- HOTA score >0.85
- No identity switches on Arno video (currently worst performer)

### **Stretch**:
- Mean error <30px (approaching annotation precision)
- >95% within 50px
- Ready for production judging at scale

---

## Risk Assessment

### **Low Risk** (Likely success):
- HOTA metric: Pure evaluation, can't break tracking
- Kalman tuning: Small change, easy to revert
- Savitzky-Golay: Post-processing, independent

### **Medium Risk** (Needs testing):
- CMC: If background features unreliable (snow, fog), may degrade performance
  - **Mitigation**: Feature quality checks, fallback to non-compensated tracking
- Global association: Complex linkage logic could introduce new errors
  - **Mitigation**: Thorough testing on unannotated videos

### **High Risk** (Significant investment):
- Pose Re-ID: Major pipeline restructuring, 3-10× compute cost
  - **Mitigation**: Only pursue if Phase 2-3 insufficient

---

## References

- **ByteTrack**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (ECCV 2022)
- **BoT-SORT**: Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracking" (arXiv 2022)
- **HOTA**: Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking" (IJCV 2021)
- **Global Association**: Fischer et al., "Global Tracking Transformers" (CVPR 2022)
