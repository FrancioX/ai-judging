# Plan: Athlete Velocity Extraction from Monocular Video

**TL;DR**: Add a new `velocity` pipeline stage that estimates the athlete's **ground-relative velocity** by subtracting camera motion (estimated via background optical flow on dark rock/tree features) from the tracked pixel velocity. Outputs per-frame velocity vectors, smoothed speed profile, trajectory smoothness, jump intervals, and an overlay visualization video.

---

### Steps

**Phase 1 — Shared Utility (background flow)**

1. **Refactor background OF functions into `src/utils/`** — move `_background_flow()`, `_dark_feature_mask()`, and `compute_cumulative_background_flow()` from `scripts/venue_camflow.py` into a new `src/utils/optical_flow.py`. Update `venue_camflow.py` to import from there. These are the proven dark-feature masking functions (target rocks/trees, exclude snow/sky) already validated at 28.5px venue accuracy.

**Phase 2 — Velocity Stage Module**

2. **Create `src/velocity/__init__.py`** (empty) + **`src/velocity/velocity.py`** with public function `extract_velocity(tracking_dir, output_dir, *, frame_root, **config) → Path`
   - Loads `tracking.json` for bbox centers per frame, loads frames from `output/frames/`
   - Per consecutive frame pair: computes background flow (camera pan), then:
     $$v_{\text{ground}} = v_{\text{athlete\_pixel}} - v_{\text{camera\_pan}} = v_{\text{athlete\_pixel}} + v_{\text{background\_flow}}$$
   - Smooths compensated velocity (configurable window)
   - Optionally normalizes speed by bbox height (distance proxy for scale invariance)

3. **Jump detection** (simple, threshold-based) — within same module or `src/velocity/jumps.py`:
   - Detect sustained upward vertical velocity (launch) → downward → stabilization (landing)
   - Outputs jump intervals: `[start_frame, end_frame, peak_frame, airtime_s]`

4. **Trajectory smoothness** — compute acceleration → jerk → smoothness metric (inverse of mean |jerk|), plus direction curvature

5. **Output**: `output/velocity/<stem>/velocity.json` — per-frame camera flow, raw velocity, compensated velocity, speed, acceleration, in_jump flag; plus summary stats and jump list

**Phase 3 — Visualization**

6. **Create `src/velocity/visualize.py`** — renders `velocity_overlay.mp4` with:
   - Speed gauge/bar (color-coded), velocity arrow on athlete, jump interval highlighting, speed timeline graph
   - Reuses rendering patterns from `src/tracking/overlay_gt.py`

**Phase 4 — Pipeline Integration**

7. **Register `"velocity"` in `src/pipeline.py`** STAGES dict — dependencies: `["tracking"]`, output_dir: `"velocity"`. Add dispatch in `_run_stage()` and validation in `_check_dependencies()`. Runs parallel to `pose_2d` in the DAG (no pose dependency).

8. **Add `velocity:` config section** to `config.yaml` — `dark_threshold`, `min_gradient`, `smooth_window`, `zoom_correct`, `normalize_by_height`, `jump_min_airtime_frames`, `jump_vy_threshold`

---

### Relevant Files
- `scripts/venue_camflow.py` — source of background OF functions to refactor (lines 96–220: `_dark_feature_mask`, `_background_flow`, `compute_cumulative_background_flow`)
- `src/pipeline.py` — STAGES registry (line 48), `_run_stage()` (line 123), `_check_dependencies()` (line 229)
- `src/tracking/overlay_gt.py` — visualization rendering patterns to reuse
- `config.yaml` — add velocity config section

### Verification
1. `uv run python -m src.pipeline "raw_videos/Ski Men_2_89_Andreas Bakke.mp4" --stage velocity` — runs without errors
2. Inspect `velocity.json` — compensated speeds are positive, direction roughly downhill
3. Watch `velocity_overlay.mp4` — speed gauge correlates with visual athlete speed
4. Run on a video with known jumps — verify detected jump intervals are visually plausible
5. `uv run ruff check src/velocity/` + `uv run pytest` — clean

### Decisions
- **No absolute/metric velocity** — all outputs in pixel space (optionally height-normalized). Metric calibration needs camera intrinsics we don't have.
- **Background OF approach** over ORB/ECC homography — simpler, proven in this venue context
- **Jump detection starts simple** — vertical velocity threshold only; can later incorporate bbox scale change or pose cues
- **Scoring integration out of scope** — velocity features can be concatenated with pose features later
- **Farneback OF cost** ~20ms/frame (~24s per 1200-frame video) — acceptable; background flow could be cached for reuse

### Further Considerations
1. **Bbox height normalization noise** — bbox detection jitter propagates into normalized speed. May want to smooth bbox height independently before dividing.
2. **Camera rotation** — current approach assumes translational pan. Median flow is fairly robust to mild rotation, but significant rotation would need per-region flow decomposition.
3. **Background flow caching** — since venue_mapping might also need the same per-frame camera flow, consider caching it to `output/velocity/<stem>/` and letting venue_mapping read from there (or vice versa).
