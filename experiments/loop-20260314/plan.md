# Experiment Loop Plan — 2026-03-14

**Branch:** `exp-loop-20260314`
**Focus:** Tracking stage — continue improving mean error and HOTA

## Baseline

| Target | Mean err (px) | HOTA | Speed (ms/frame) |
|--------|:---:|:---:|:---:|
| Accuracy-best (Exp 21a + loop-20260313 Iter 5) | 26.2 (dev-10) | 0.867 | ~45.0 |
| Speed-best (Exp 15, adaptive threshold no-OF) | 25.5 (18-video) | 0.856 | 6.05 |

Mini-set (4 videos) baseline: **26.60px / 0.844 HOTA**

Primary failure cases: Arno Vuarnier (~47.9px), Quentin Puydenus (~47.8px), Gabin Leonard (~58.5px), Theodor Salen (~54.6px).

## Candidate List (priority order)

1. **CoTracker3 lower `min_visible` threshold** — infrastructure exists; Iter 4 failed with 0.5 (all fell back to LK); try 0.1–0.3 to allow partial aerial tracking. Fast (0.5 days).
2. **Savitzky-Golay smoothing for short gap interpolation** — replace linear interpolation for short gaps (<10 frames) with polynomial smoothing. Fast (0.5–1 day). Note: not expected to help aerial cases but may help short-gap accuracy.
3. **Pose plausibility scoring (YOLO-Pose on Phase A candidates)** — inline pose inference on conflict candidates; bystanders standing still or facing away have distinct keypoints from active skier. Medium effort (2–3 days).
4. **OSNet / FastReID appearance embeddings** — trained appearance signal to break the Gabin Leonard self-reinforcing bystander feedback loop. Heavy (3–5 days).
5. **SAMURAI for aerial-gap recovery** — SAM2-based zero-shot tracking for long aerial phases; highest ceiling. Heaviest effort (5–7 days).
6. **Global tracklet association (Hungarian / min-cost flow)** — offline bidirectional pass for fragment merging. Medium-heavy (5–7 days).

## Strategy

Start with the fastest candidates (1, 2) for quick wins. Move to medium candidates (3) before committing to the heavy ones (4, 5, 6). Accept only if mini-set improves; confirm on dev-set before declaring accuracy-best.
