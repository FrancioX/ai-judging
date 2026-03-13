# Tracking Experiment Series 01 — Summary

**Period:** 2026-02-27 → 2026-03-13
**Scope:** 21 experiments (+ sub-iterations) improving the temporal tracking stage of the freeride skiing pipeline.
**Evaluation:** 18 annotated videos (ski + snowboard), GT sparse center-point annotations interpolated during eval. Corrected annotations applied 2026-03-12.
**Metrics:** Mean center error (px), HOTA (identity), ms/frame (tracking stage only, not YOLO segmentation).

---

## Final Results at a Glance

| Config | 18-video mean error | HOTA | ms/frame | Use when |
|--------|:-------------------:|:----:|:--------:|:---------|
| **Best overall — Exp 21a** | **22.4 px** | **0.876** | **41.9** | Quality-first |
| **Best lightweight — Exp 15** | **25.5 px** | **0.856** | **6.05** | Speed-first (~7× faster) |
| Original baseline | ~39.5 px* | ~0.836* | — | — |

*Baseline measured on 3 videos only; not re-validated on full 18-video set.

**Total improvement vs baseline (primary 3 videos):** −45% mean error (39.5 → 21.9 px), +4.8% HOTA (0.836 → 0.876).

---

## Best Config — Overall (Exp 21a)

```yaml
tracking:
  optical_flow_method: "auto"
  of_trace_filter_enabled: false   # trace as soft score only — no hard proximity gate
  of_gap_fill_enabled: true
  of_min_gap_for_fill: 10          # gaps < 10 frames use linear interpolation
  of_drift_guard_px: 200.0         # fall back to linear if OF drifts > 200px
  of_reanchor_min_conf: 0.0
  merge_threshold_adaptive: true
  merge_threshold_low: 0.5
  merge_threshold_high: 0.6
  merge_threshold_min_overlap_ratio: 0.55
  identity_guard_enabled: false
  smooth_window: 5
  w_continuity: 0.6
  w_track_stickiness: 0.4
  w_velocity: 0.4
  of_synthetic_confidence: 0.3
```

## Best Config — Lightweight (Exp 15)

```yaml
tracking:
  optical_flow_method: "none"
  merge_threshold_adaptive: true
  merge_threshold_low: 0.5
  merge_threshold_high: 0.6
  merge_threshold_min_overlap_ratio: 0.55
  identity_guard_enabled: false
  smooth_window: 5
  w_continuity: 0.6
  w_track_stickiness: 0.4
```

---

## Experiment Chronicle

Each experiment is summarised as: goal → outcome → why.

### Experiments 1–5: OF-Era (with hard proximity gate active)

| Exp | Goal | Outcome | Why |
|-----|------|---------|-----|
| **1** | Velocity-consistency score for conflict resolution | Neutral (kept at w=0.4) | Signal too weak to flip decisions; higher weights poison velocity history via wrong picks |
| **2a** | Inject synthetic OF candidate when skier undetected mid-air | Catastrophic regression | Synthetics received full OF agreement bonus — tautologically perfect, dominated scoring |
| **2b** | Fix: inject only when all real candidates fail OF gate + zero OF score for synthetics | **+33% error reduction on Arno, first major win** | Correct: synthetics plugged aerial-phase bystander takeover without circular self-reinforcement |
| **3** | Camera-motion-compensated velocity | Regression | CMC subtraction had no differentiating effect; regressions were from higher w_velocity, not CMC noise |
| **4** | Instrument Phase A.6 identity guard jump distributions | No change (pure diagnostics) | Revealed bimodal gap: p90 < 40px vs rejected median ~400px — threshold at 150px was in a dead zone |
| **5** | Sweep `max_jump_px` (10/20/50/100) | **`max_jump_px=20` optimal** — −0.6% error, +0.4% AssA | 20 sits between normal motion (p90 < 40px) and identity switches (rejected median 370–660px) |

### Experiments 6–15: No-OF Track Cleaning

The key insight from Exp 6 onwards: **the wrong-person candidates enter through low-scoring merged tracks, not through OF failure**. Cleaning the merge pool (raising `merge_score_threshold`) had far more impact than any scoring weight tuning.

| Exp | Goal | Outcome | Why |
|-----|------|---------|-----|
| **6** | Validate no-OF setup + Kalman smoothing | Baseline: 91.8 px mean error @ 6.4ms/frame | Showed magnitude of OF contribution and fast-iteration benchmark |
| **7** | No-OF identity guard (prev-detection distance only) | Major regression (90 → 198.9 px) | Without OF to keep trace on-track, prev-det guard over-rejects or locks wrong trajectory |
| **8** | Raise `merge_score_threshold` 0.3 → 0.5 | **−46% error** (first clean no-OF win) | Cutting low-scoring bystander tracks from the merge pool directly reduces wrong-person candidates |
| **9** | Boost `w_continuity` + `w_track_stickiness` | No change | With a cleaner merge pool, scoring weights were no longer the bottleneck |
| **10** | Filter low-confidence detections from merge pool | Regression on Lach | Too aggressive — removed valid low-confidence skier detections, increased interpolation burden |
| **11** | Sweep `merge_score_threshold` 0.45–0.70 | **`0.60` optimal** — 24.9 px / 0.855 HOTA (18-video) | Phase transition at 0.55–0.60: one specific bystander track scores in that range and dominates when included |
| **12** | Re-add OF on top of threshold=0.6 | Regression | OF gap-fill drifts onto bystanders when merge pool is small; two optima exist but they interfere |
| **13** | Candidate pool diagnostics (instrumentation) | No change, tooling only | Added `threshold_sensitivity_tracks`, `conflict_summary` to tracking manifest for evidence-driven tuning |
| **14** | Diagnose Arno's regression at threshold=0.6 | Root cause: track #70 (10 dets, 85px from GT) excluded | Identified single skier fragment lost at 0.6 that caused Arno's +12px vs 0.5 |
| **15** | Adaptive threshold: auto-select 0.5 vs 0.6 per video | **25.5 px / 0.856 HOTA (18-video, 6.05 ms/frame)** — **Best lightweight** | `det_rate < 0.55` → skier fragmented → use 0.5; otherwise 0.6 cuts bystanders correctly |

### Experiments 16–18: Failed Lightweight OF Attempts

These three experiments targeted specific OF sub-components while leaving the hard proximity gate active. All failed for the same structural reason (see architecture section below).

| Exp | Goal | Outcome | Why it failed |
|-----|------|---------|---------------|
| **16** | OF trace for Phase A scoring only, no Phase B gap-fill | Catastrophic regression | Trace drifted without gap-fill → proximity gate discarded valid skier candidates |
| **17** | Conditional gap-fill: only gaps ≥ 10 frames use OF | Neutral | Drift guard worked (no regression) but conditions where OF genuinely helped were too rare with gate active |
| **18** | High-confidence re-anchoring (min_conf=0.7) | Regression | Too strict — many valid skier detections have confidence 0.5–0.7; drifted trace then misfired the gate |

### Experiments 19–21: Architectural Fix + Soft Trace

| Exp | Goal | Outcome | Why |
|-----|------|---------|-----|
| **19** | Add `of_trace_filter_enabled` flag — decouple Phase A hard gate from scoring | No change (pure infrastructure) | Enables safe re-testing of Exps 16–18 without drift→gate cascade |
| **20** | Test soft-trace + full gap-fill (`of_trace_filter_enabled=false`) | **23.5 px / 0.869 HOTA, ~45.2 ms/frame** — beats no-OF by −7.8% error | With gate removed, trace score signal contributes positively; gap-fill keeps trace accurate |
| **20** (also) | Test soft-trace, no gap-fill | Identical to no-OF baseline | Without gap-fill trace drifts → score signal becomes noise; gap-fill is essential |
| **21a** | Conditional gap-fill with filter disabled (`of_min_gap_for_fill=10`, `of_drift_guard_px=200`) | **22.4 px / 0.876 HOTA, 41.9 ms/frame** — **Best overall** | Short gaps (<10 frames) use linear interpolation — removes short-gap OF noise, faster, no regression |
| **21b** | High-conf re-anchoring with filter disabled | Neutral | Without the proximity gate, a drifted trace no longer misfires; re-anchoring has nothing to fix |
| **21c** | Conditional gap-fill + high-conf re-anchoring | Same as 21a | Re-anchoring adds nothing on top of conditional gap-fill |

---

## Key Architectural Insights

### 1. The Phase A/B Coupling Problem

The original OF design had a structural coupling that made it fragile:

- **Phase A** (`_resolve_merge_conflicts._score_pass`): OF trace used for *two* purposes — soft `w_of_agreement` score AND hard proximity gate (discard candidates > 2×150px from trace).
- **Phase B** (`_fill_gaps_optical_flow`): Gap-fill keeps Phase A trace anchored by filling detection gaps first.

**Consequence:** Any intervention degrading trace quality (disabling gap-fill, restricting re-anchoring, drift guards) caused the proximity gate to misfire → fewer detected frames → more interpolation errors. The OF system was coherent only when all parts ran together. This is why Exps 16–18 all failed until Exp 19 decoupled the gate.

### 2. Merge Pool Quality > Scoring Weights

Experiments 8–15 showed that the dominant failure mode in no-OF mode was **low-scoring bystander tracks entering the merge candidate pool**, not wrong scoring decisions. Once the pool was cleaned (`merge_score_threshold` tuning), scoring weight changes (Exp 9) had zero effect. Lesson: fix the input distribution before tuning the scoring function.

### 3. Bimodal Jump Distribution

Phase A.6 identity guard diagnostics (Exp 4) revealed a clean bimodal structure: inter-frame distances cluster at p90 < 40px (normal motion) and ~400–660px (identity switches). There is almost nothing in between. `max_jump_px=20` exploits this gap cleanly.

### 4. Velocity Signal is a Dead End (at current architecture)

The velocity-consistency signal (Exp 1) cannot be raised above `w=0.4` because wrong candidate picks poison the velocity history, which then reinforces more wrong picks (positive feedback). Camera-motion compensation (Exp 3) had no effect because CMC is not the bottleneck — the feedback loop is.

---

## What Was Not Tried (Future Work)

| Idea | Expected impact | Effort | Notes |
|------|:---------------:|:------:|-------|
| **ReID embeddings (OSNet-x0.25)** | High | Medium | 128-dim appearance embeddings replace spatial continuity in conflict resolution. Direct fix for skier-vs-bystander at representation level. Hardest remaining videos (Arno 47.9px, Quentin 47.8px) likely need this. |
| **Learned optical flow (RAFT)** | Medium–High | Medium | Far more accurate than LK on snow/sky textures and large motion. Drop-in for `_fill_gaps_optical_flow()` and `_build_of_trace()`. ~10–30s/video on MPS. |
| **Per-gap anchor quality check** | Medium | Small | Before invoking OF on a gap, verify both anchor detections are high-confidence (>0.6). More principled version of `of_min_gap_for_fill`. |
| **Sweep `of_min_gap_for_fill`** | Small | Tiny | Current value of 10 was not swept — values 5, 15, 20 not tested. Easy win candidate. |
| **Temporal attention / global track selection** | High | Large | Replace greedy forward merge with joint scoring over full video. Likely overkill; revisit only if ReID+OF cannot close the gap. |

---

## Evaluation Infrastructure Built

| Tool | Purpose |
|------|---------|
| `src/tracking/evaluate.py --batch` | HOTA + mean error across all 18 annotated videos |
| `src/tracking/annotate_centers` | Ground-truth annotation tool (center-point, sparse) |
| `src/tracking/overlay_gt` | Visual overlay: green=GT, red=predicted |
| `scripts/sweep_max_jump.py` | Sweep `identity_guard_max_jump_px` |
| `scripts/sweep_merge_threshold.py` | Sweep `merge_score_threshold` |
| `scripts/sweep_of_jump.py` | Sweep OF + identity guard combinations |
| `scripts/sweep_trace_filter.py` | Sweep `of_trace_filter_enabled` variants |
| `scripts/sweep_exp21.py` | Sweep conditional gap-fill + re-anchoring |
| `scripts/batch_track_and_eval.py` | Full 18-video track + eval from existing segmentation manifests |
| `scripts/time_no_of_tracking.py` | Time no-OF tracking to temp dir (no output overwrite) |
| `scripts/diagnose_arno_threshold.py` | Diagnose which tracks are lost at threshold boundaries |
