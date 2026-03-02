"""Temporal skier tracking — best-track selection, optical-flow gap filling and
velocity-aware bbox smoothing.

Reads the per-frame segmentation manifest (with all detected persons and
ByteTrack track IDs) and produces a refined tracking manifest where:

1. The skier track is selected via a weighted composite score
   (confidence × center-proximity × track-length).
2. **Every** frame is guaranteed to have exactly one bounding box.
   Detection gaps are filled using optical-flow-guided motion estimation
   (sparse Lucas-Kanade with dense Farneback fallback). When optical flow
   is disabled or unavailable, falls back to linear interpolation.
3. Bounding boxes are temporally smoothed with a velocity-aware
   bidirectional filter using per-frame flow velocities.
4. New crops are written using the refined bounding boxes.

This stage sits between segmentation and 2D-pose estimation and follows
the project convention of one public function per module returning a Path.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import cv2
from tqdm import tqdm

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def track_skier(
    seg_manifest_path: str | Path,
    frame_dir: str | Path,
    output_dir: str | Path,
    *,
    w_conf: float = 0.3,
    w_center: float = 0.5,
    w_length: float = 0.2,
    min_track_frames: int = 10,
    smooth_window: int = 5,
    padding_ratio: float = 0.15,
    merge_tracks: bool = True,
    merge_score_threshold: float = 0.3,
    optical_flow_method: str = "auto",
    flow_max_extrapolate_frames: int = 30,
    flow_min_keypoints: int = 5,
    identity_guard_enabled: bool = True,
    identity_guard_max_jump_px: float = 150.0,
    identity_guard_reanchor_interval: int = 50,
    identity_guard_reanchor_min_conf: float = 0.5,
    identity_guard_max_drift_px: float = 200.0,
    w_velocity: float = 0.4,
    vel_history_len: int = 5,
    of_synthetic_confidence: float = 0.3,
    cmc_enabled: bool = True,
    cmc_method: str = "orb",
    cmc_exclude_margin: float = 1.5,
    cmc_min_features: int = 20,
    cmc_ransac_threshold: float = 3.0,
) -> Path:
    """Select the best skier track(s), smooth bboxes, fill gaps and re-crop.

    Every frame is guaranteed to have exactly one bounding box after this
    stage — detection gaps are filled using optical-flow-guided motion
    estimation (when enabled) or linear interpolation as a fallback.

    When merge_tracks=True, multiple high-scoring tracks are merged to handle
    tracking fragmentation (same skier split across multiple track IDs).

    Parameters
    ----------
    seg_manifest_path : path to the segmentation.json produced by YOLO-Seg.
    frame_dir : directory containing the original extracted frames.
    output_dir : base output directory for tracking results.
    w_conf : weight for mean detection confidence in track scoring.
    w_center : weight for center-proximity in track scoring.
    w_length : weight for track duration in track scoring.
    min_track_frames : ignore tracks shorter than this.
    smooth_window : EMA / moving-average window (0 = disabled).
    padding_ratio : extra padding around bbox (fraction of width/height).
    merge_tracks : if True, merge multiple high-scoring tracks.
    merge_score_threshold : minimum score for track inclusion in merge.
    optical_flow_method : ``"auto"`` (sparse LK, fallback to dense),
        ``"sparse"`` (Lucas-Kanade only), ``"dense"`` (Farneback only),
        or ``"none"`` (disable optical flow, use linear interpolation).
    flow_max_extrapolate_frames : max frames to extrapolate via optical
        flow for leading/trailing gaps (beyond this, copy anchor).
    flow_min_keypoints : minimum tracked keypoints before sparse LK
        falls back to dense Farneback (used when method is ``"auto"``).
    identity_guard_enabled : if True, use OF-based identity guard to reject
        potential identity switches.
    identity_guard_max_jump_px : reject detections >N px from OF prediction
        AND >N/2 px from previous detection.
    identity_guard_reanchor_interval : re-anchor OF trace every N confident
        detected frames.
    identity_guard_reanchor_min_conf : minimum detection confidence for
        re-anchor.
    identity_guard_max_drift_px : force re-anchor if OF drift since last
        anchor exceeds this threshold.
    cmc_enabled : if True, enable camera motion compensation to separate
        camera motion from object motion in optical flow.
    cmc_method : ``"orb"`` (ORB features, fastest) or ``"ecc"``
        (Enhanced Correlation Coefficient, more robust) or ``"none"``.
    cmc_exclude_margin : multiplier for skier bbox exclusion zone
        (1.5 = exclude 1.5× bbox area from feature matching).
    cmc_min_features : minimum matched features required for valid
        camera motion estimate (fallback to no compensation if lower).
    cmc_ransac_threshold : pixel threshold for RANSAC outlier rejection
        in homography estimation.

    Returns
    -------
    Path to the tracking manifest JSON (``tracking.json``).
    """
    seg_manifest_path = Path(seg_manifest_path)
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    with open(seg_manifest_path) as f:
        seg_data = json.load(f)

    seg_frames: list[dict] = seg_data.get("frames", [])
    n_frames = len(seg_frames)
    if n_frames == 0:
        raise FileNotFoundError("Segmentation manifest contains no frames")

    # Infer image dimensions from the first frame
    sample_frame_file = seg_frames[0].get("frame_file", "")
    sample_img = cv2.imread(str(frame_dir / sample_frame_file))
    if sample_img is None:
        raise FileNotFoundError(
            f"Cannot read frame: {frame_dir / sample_frame_file}"
        )
    img_h, img_w = sample_img.shape[:2]

    print(f"Running temporal tracking on {n_frames} frames …")
    print(f"  Weights — conf: {w_conf}  center: {w_center}  length: {w_length}")
    print(f"  Smooth window: {smooth_window}  |  Full coverage: guaranteed")
    if merge_tracks:
        print(f"  Multi-track merge: enabled (threshold {merge_score_threshold:.2f})")

    # ------------------------------------------------------------------
    # Phase A — Build per-track history and score each track
    # ------------------------------------------------------------------
    tracks: dict[int, list[dict]] = {}  # track_id → list of observations

    for fr in seg_frames:
        for p in fr.get("persons", []):
            tid = p.get("track_id")
            if tid is None:
                continue
            tracks.setdefault(tid, []).append({
                "frame_id": fr["frame_id"],
                "bbox": p["bbox"],
                "confidence": p["confidence"],
                "area": p["area"],
                "track_id": tid,
            })

    if not tracks:
        # Fallback: no track IDs available — use the per-frame selected
        # person from the segmentation manifest directly.
        print("  ⚠ No track IDs found — falling back to segmentation selection")
        return _write_passthrough(seg_data, frame_dir, output_dir,
                                  padding_ratio, crop_dir)

    max_dist = math.sqrt((img_w / 2) ** 2 + (img_h / 2) ** 2)
    cx, cy = img_w / 2, img_h / 2

    # Score all tracks
    track_scores: list[tuple[int, float, int]] = []  # (track_id, score, n_detections)

    for tid, obs in tracks.items():
        if len(obs) < min_track_frames:
            continue

        mean_conf = sum(o["confidence"] for o in obs) / len(obs)

        # Center proximity: 1.0 = perfect center, 0.0 = corner
        center_scores: list[float] = []
        for o in obs:
            bx = (o["bbox"][0] + o["bbox"][2]) / 2
            by = (o["bbox"][1] + o["bbox"][3]) / 2
            dist = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
            center_scores.append(1.0 - dist / max_dist)
        mean_center = sum(center_scores) / len(center_scores)

        length_ratio = len(obs) / n_frames

        score = (w_conf * mean_conf
                 + w_center * mean_center
                 + w_length * length_ratio)

        track_scores.append((tid, score, len(obs)))

    if not track_scores:
        # All tracks were too short — pick the longest one
        best_tid = max(tracks, key=lambda t: len(tracks[t]))
        track_scores = [(best_tid, 0.0, len(tracks[best_tid]))]

    # Sort by score descending
    track_scores.sort(key=lambda x: x[1], reverse=True)

    # Select track(s) based on merge strategy
    if merge_tracks:
        # Select all tracks that meet the score threshold
        selected_tracks = [
            tid for tid, score, _ in track_scores
            if score >= merge_score_threshold
        ]
        if not selected_tracks:
            # Fallback: use the best track even if below threshold
            selected_tracks = [track_scores[0][0]]

        total_detections = sum(len(tracks[tid]) for tid in selected_tracks)
        print(f"  → Merging {len(selected_tracks)} track(s): {selected_tracks[:5]}{'...' if len(selected_tracks) > 5 else ''}")
        print(f"  → Combined detections: {total_detections}")

        # Merge observations from all selected tracks.
        # Collect ALL candidates per frame, then resolve conflicts with a
        # forward continuity pass that penalises large spatial jumps.
        frame_candidates: dict[int, list[dict]] = {}
        for tid in selected_tracks:
            for o in tracks[tid]:
                fid = o["frame_id"]
                frame_candidates.setdefault(fid, []).append(o)

        # Build a preliminary OF trace using ALL candidates for re-anchoring.
        # Unlike single-track seeding, this follows the skier across track
        # boundaries by re-anchoring to the closest candidate within range.
        preliminary_of_trace: dict[int, tuple[float, float]] = {}
        use_flow_for_merge = (
            optical_flow_method != "none"
            and _HAS_NUMPY
        )
        if use_flow_for_merge:
            # Seed from the best track's first detection
            best_track_obs_list = tracks[track_scores[0][0]]
            seed_obs = best_track_obs_list[0] if best_track_obs_list else None
            print(f"  Building multi-candidate OF trace (seed: track "
                  f"{track_scores[0][0]}, {len(frame_candidates)} frames with detections)…")
            preliminary_of_trace = _build_multi_candidate_of_trace(
                frame_candidates, n_frames, img_w, img_h,
                frame_dir, seg_frames,
                seed_obs=seed_obs,
                method=optical_flow_method,
                min_keypoints=flow_min_keypoints,
                max_reanchor_dist_px=150.0,
                reanchor_interval=5,
                max_drift_px=150.0,
            )
            print(f"  → Preliminary OF trace: {len(preliminary_of_trace)} frames")

        selected_obs = _resolve_merge_conflicts(
            frame_candidates, cx, cy, max_dist, w_conf, w_center,
            w_velocity=w_velocity,
            vel_history_len=vel_history_len,
            of_synthetic_confidence=of_synthetic_confidence,
            frame_dir=frame_dir,
            seg_frames=seg_frames,
            optical_flow_method=optical_flow_method,
            flow_min_keypoints=flow_min_keypoints,
            of_trace=preliminary_of_trace or None,
        )

        # Store metadata for manifest (use top track as representative)
        best_tid = selected_tracks
        best_score = track_scores[0][1]
    else:
        # Single track selection (original behavior)
        best_tid, best_score, best_count = track_scores[0]
        selected_obs = {o["frame_id"]: o for o in tracks[best_tid]}
        print(f"  → Selected track {best_tid} "
              f"({best_count} detections, score={best_score:.3f})")

    # ------------------------------------------------------------------
    # Phase A.5 — Check if optical flow is available
    # ------------------------------------------------------------------
    use_flow = (
        optical_flow_method != "none"
        and _HAS_NUMPY
    )

    # ------------------------------------------------------------------
    # Phase A.6 — Identity guard: OF trace + rejection
    # ------------------------------------------------------------------
    of_trace: dict[int, tuple[float, float]] = {}
    identity_rejections = 0
    identity_jump_stats: dict[str, object] | None = None

    if identity_guard_enabled and use_flow:
        print(f"  Identity guard: enabled (max_jump={identity_guard_max_jump_px}px, "
              f"reanchor_interval={identity_guard_reanchor_interval})")
        of_trace = _maintain_of_trace(
            selected_obs, n_frames, img_w, img_h,
            frame_dir, seg_frames,
            method=optical_flow_method,
            min_keypoints=flow_min_keypoints,
            reanchor_interval=identity_guard_reanchor_interval,
            reanchor_min_conf=identity_guard_reanchor_min_conf,
            max_drift_px=identity_guard_max_drift_px,
        )

        selected_obs, identity_meta = _validate_identity(
            selected_obs, of_trace,
            max_jump_px=identity_guard_max_jump_px,
        )
        identity_rejections = identity_meta.get("rejections", 0)
        raw_jump_stats = identity_meta.get("jump_stats")
        if isinstance(raw_jump_stats, dict):
            identity_jump_stats = raw_jump_stats

        if identity_rejections > 0:
            print(f"  → Identity guard: rejected {identity_rejections} detections")
        if identity_jump_stats is not None:
            of_stats = identity_jump_stats.get("of_dist_px")
            if isinstance(of_stats, dict) and of_stats.get("count", 0) > 0:
                over_max = int(identity_jump_stats.get("over_max_jump_count", 0))
                print(
                    "  → Identity jumps vs OF (px): "
                    f"mean={of_stats['mean']}, p90={of_stats['p90']}, "
                    f"max={of_stats['max']} "
                    f"(>max_jump: {over_max}/{of_stats['count']})",
                )
            prev_stats = identity_jump_stats.get("prev_det_dist_px")
            if isinstance(prev_stats, dict) and prev_stats.get("count", 0) > 0:
                over_prev = int(identity_jump_stats.get("over_prev_jump_count", 0))
                prev_threshold = identity_jump_stats.get(
                    "prev_jump_threshold_px",
                    round(identity_guard_max_jump_px * 0.5, 2),
                )
                print(
                    "  → Identity jumps vs prev det (px): "
                    f"mean={prev_stats['mean']}, p90={prev_stats['p90']}, "
                    f"max={prev_stats['max']} "
                    f"(>{prev_threshold}: {over_prev}/{prev_stats['count']})",
                )
    elif identity_guard_enabled and not use_flow:
        print("  ⚠ Identity guard enabled but optical flow disabled — skipping")

    # ------------------------------------------------------------------
    # Phase B — Assemble per-frame bboxes: detected → interpolated → extrapolated
    # ------------------------------------------------------------------
    frame_bboxes: dict[int, dict] = {}  # frame_id → {bbox, confidence, source}

    for fid in range(n_frames):
        if fid in selected_obs:
            o = selected_obs[fid]
            is_synthetic = o.get("of_synthetic", False)
            frame_bboxes[fid] = {
                "bbox": o["bbox"],
                "confidence": o["confidence"],
                "detected": not is_synthetic,
                "interpolated": is_synthetic,
                "of_synthetic": is_synthetic,
                "of_predicted": of_trace.get(fid),
            }

    # ------------------------------------------------------------------
    # Phase B — Fill ALL gaps so every frame has exactly one bbox
    # ------------------------------------------------------------------
    flow_velocities: dict[int, tuple[float, float]] = {}
    of_method_used = "none"

    if use_flow:
        print(f"  Optical flow: {optical_flow_method} "
              f"(min_kp={flow_min_keypoints}, max_extrap={flow_max_extrapolate_frames})")
        if cmc_enabled:
            print(f"  Camera motion compensation: enabled (method={cmc_method}, "
                  f"min_features={cmc_min_features})")
        flow_velocities, of_method_used = _fill_gaps_optical_flow(
            frame_bboxes, n_frames, img_w, img_h,
            frame_dir, seg_frames,
            method=optical_flow_method,
            min_keypoints=flow_min_keypoints,
            max_extrapolate=flow_max_extrapolate_frames,
            cmc_enabled=cmc_enabled,
            cmc_method=cmc_method,
            cmc_exclude_margin=cmc_exclude_margin,
            cmc_min_features=cmc_min_features,
            cmc_ransac_threshold=cmc_ransac_threshold,
        )
    else:
        if optical_flow_method != "none":
            print("  ⚠ numpy not available — falling back to linear interpolation")
        _fill_gaps(frame_bboxes, n_frames, img_w, img_h)

    assert len(frame_bboxes) == n_frames, (
        f"Full coverage failed: {len(frame_bboxes)}/{n_frames} frames have bboxes"
    )

    # ------------------------------------------------------------------
    # Phase C — Temporal bbox smoothing
    # ------------------------------------------------------------------
    smoothed_velocities: dict[int, tuple[float, float]] = {}
    if smooth_window > 0 and len(frame_bboxes) > 1:
        if use_flow and flow_velocities:
            smoothed_velocities = _smooth_bboxes_velocity_aware(
                frame_bboxes, n_frames, smooth_window, flow_velocities,
            )
        else:
            _smooth_bboxes(frame_bboxes, n_frames, smooth_window)

    # ------------------------------------------------------------------
    # Phase D — Apply padding and write crops + manifest
    # ------------------------------------------------------------------
    manifest_frames: list[dict] = []

    for idx in tqdm(range(n_frames), desc="Tracking crops"):
        seg_fr = seg_frames[idx]
        frame_file = seg_fr["frame_file"]
        fb = frame_bboxes[idx]

        x1, y1, x2, y2 = fb["bbox"]
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(img_w, x2 + pad_x)
        py2 = min(img_h, y2 + pad_y)

        img = cv2.imread(str(frame_dir / frame_file))
        if img is None:
            continue
        crop = img[py1:py2, px1:px2]
        crop_name = f"crop_{idx:06d}.jpg"
        cv2.imwrite(str(crop_dir / crop_name), crop)

        manifest_frames.append({
            "frame_id": idx,
            "frame_file": frame_file,
            "detected": fb["detected"],
            "interpolated": fb["interpolated"],
            "of_synthetic": fb.get("of_synthetic", False),
            "bbox": [x1, y1, x2, y2],
            "bbox_padded": [px1, py1, px2, py2],
            "confidence": fb["confidence"],
            "track_id": best_tid if isinstance(best_tid, int) else -1,  # -1 for merged tracks
            "crop_file": crop_name,
            "flow_displacement": (
                [round(flow_velocities[idx][0], 2),
                 round(flow_velocities[idx][1], 2)]
                if idx in flow_velocities else None
            ),
            "of_predicted": (
                [round(fb["of_predicted"][0], 2), round(fb["of_predicted"][1], 2)]
                if fb.get("of_predicted") is not None else None
            ),
            "velocity": (
                [round(smoothed_velocities[idx][0], 2),
                 round(smoothed_velocities[idx][1], 2)]
                if idx in smoothed_velocities else None
            ),
            "speed_px_per_frame": (
                round(math.sqrt(
                    smoothed_velocities[idx][0] ** 2
                    + smoothed_velocities[idx][1] ** 2
                ), 2)
                if idx in smoothed_velocities else None
            ),
        })

    # Compute velocity summary stats
    velocity_stats: dict[str, float] | None = None
    if smoothed_velocities:
        speeds = [
            math.sqrt(vx ** 2 + vy ** 2)
            for vx, vy in smoothed_velocities.values()
        ]
        vys = [vy for _, vy in smoothed_velocities.values()]
        velocity_stats = {
            "mean_speed": round(sum(speeds) / len(speeds), 2),
            "max_speed": round(max(speeds), 2),
            "mean_vy": round(sum(vys) / len(vys), 2),
        }

    # Write manifest
    manifest = {
        "source_segmentation": str(seg_manifest_path),
        "selected_track_id": best_tid if isinstance(best_tid, int) else best_tid[0],
        "merged_tracks": best_tid if isinstance(best_tid, list) else None,
        "track_score": round(best_score, 4),
        "weights": {"w_conf": w_conf, "w_center": w_center, "w_length": w_length},
        "smooth_window": smooth_window,
        "padding_ratio": padding_ratio,
        "optical_flow_method_used": of_method_used if use_flow else "none",
        "identity_guard_enabled": identity_guard_enabled,
        "identity_guard_rejections": identity_rejections,
        "identity_guard_jump_stats": identity_jump_stats,
        "velocity_stats": velocity_stats,
        "n_frames": n_frames,
        "frames": manifest_frames,
    }
    manifest_path = output_dir / "tracking.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_detected = sum(1 for fr in manifest_frames if fr["detected"])
    n_interp = sum(1 for fr in manifest_frames if fr["interpolated"])
    print(f"  → Tracked: {n_detected} detected, {n_interp} interpolated "
          f"({n_frames} total — full coverage)")
    print(f"  → Crops saved to {crop_dir}")
    print(f"  → Manifest saved to {manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _velocity_consistency(
    vel_history: "collections.deque[tuple[float, float]]",
    candidate_vx: float,
    candidate_vy: float,
    min_history: int = 3,
) -> float:
    """Score how well a candidate's implied velocity matches recent motion.

    Combines direction similarity (cosine) and magnitude consistency to
    distinguish the moving athlete from stationary bystanders or people
    moving in a different direction.

    Parameters
    ----------
    vel_history : recent ``(vx, vy)`` tuples from chosen detections.
    candidate_vx, candidate_vy : implied velocity of the candidate.
    min_history : minimum entries before scoring is active.

    Returns
    -------
    Score in ``[0, 1]``.  Returns 0.5 (neutral) during cold start.
    """
    if len(vel_history) < min_history:
        return 0.5

    # Weighted mean of recent velocities (more recent = higher weight)
    weights = [i + 1 for i in range(len(vel_history))]
    w_sum = sum(weights)
    pred_vx = sum(w * v[0] for w, v in zip(weights, vel_history)) / w_sum
    pred_vy = sum(w * v[1] for w, v in zip(weights, vel_history)) / w_sum

    pred_mag = math.sqrt(pred_vx ** 2 + pred_vy ** 2)
    cand_mag = math.sqrt(candidate_vx ** 2 + candidate_vy ** 2)

    # --- Direction similarity (cosine) ---
    eps = 1e-6
    if pred_mag < eps or cand_mag < eps:
        # One or both are near-stationary — direction undefined
        # If predicted is moving but candidate is stationary, penalise
        if pred_mag > 2.0 and cand_mag < 1.0:
            direction_sim = 0.0
        else:
            direction_sim = 0.5
    else:
        dot = pred_vx * candidate_vx + pred_vy * candidate_vy
        cosine = dot / (pred_mag * cand_mag)
        direction_sim = max(0.0, (cosine + 1.0) / 2.0)  # map [-1, 1] → [0, 1]

    # --- Magnitude consistency ---
    if pred_mag < eps:
        magnitude_sim = 1.0 if cand_mag < 2.0 else max(0.0, 1.0 - cand_mag / 50.0)
    else:
        ratio = abs(cand_mag - pred_mag) / (pred_mag + eps)
        magnitude_sim = max(0.0, 1.0 - ratio)

    return 0.65 * direction_sim + 0.35 * magnitude_sim


def _score_detection(
    detection: dict,
    cx: float,
    cy: float,
    max_dist: float,
    w_conf: float,
    w_center: float,
) -> float:
    """Score a single detection based on confidence and center proximity.

    Used when merging multiple tracks to pick the best detection per frame.
    """
    conf = detection["confidence"]

    # Center proximity
    bbox = detection["bbox"]
    bx = (bbox[0] + bbox[2]) / 2
    by = (bbox[1] + bbox[3]) / 2
    dist = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
    center_score = 1.0 - dist / max_dist

    return w_conf * conf + w_center * center_score


def _resolve_merge_conflicts(
    frame_candidates: dict[int, list[dict]],
    cx: float,
    cy: float,
    max_dist: float,
    w_conf: float,
    w_center: float,
    *,
    w_continuity: float = 0.6,
    w_track_stickiness: float = 0.4,
    w_of_agreement: float = 1.5,
    w_velocity: float = 0.4,
    vel_history_len: int = 5,
    of_tight_radius_px: float = 150.0,
    of_synthetic_confidence: float = 0.3,
    frame_dir: Path | None = None,
    seg_frames: list[dict] | None = None,
    optical_flow_method: str = "auto",
    flow_min_keypoints: int = 5,
    of_trace: dict[int, tuple[float, float]] | None = None,
) -> dict[int, dict]:
    """Pick one detection per frame using optical-flow continuity and track-ID stickiness.

    For frames with a single candidate the choice is trivial.  When
    multiple candidates exist, a forward pass blends:

    1. **Quality** — confidence + center-proximity (static per detection).
    2. **Continuity** — proximity to the *predicted* position of the
       previous bbox.  When optical flow is available the prediction uses
       the measured displacement; otherwise it falls back to the raw
       previous centre (zero-velocity assumption).
    3. **Track stickiness** — bonus for having the same ByteTrack ID as
       the previously chosen detection, preventing identity switches
       during crossover events.
    4. **OF agreement** — bonus for proximity to the OF-predicted position
       from the independent OF trace (when provided).
    5. **Velocity consistency** — bonus for candidates whose implied
       velocity (direction + magnitude) matches recent motion history.
       Distinguishes the moving athlete from stationary bystanders.
    6. **Synthetic OF candidate** — when no YOLO detection is near the
       OF-predicted position, a synthetic candidate is injected at the
       OF center using the previous bbox size.  This preserves tracking
       through aerial phases where the skier is undetected by YOLO but
       bystanders are.

    Parameters
    ----------
    frame_candidates : mapping from ``frame_id`` to a *list* of
        candidate observation dicts (each with ``bbox``, ``confidence``).
    cx, cy : image center coordinates.
    max_dist : diagonal distance used for normalisation.
    w_conf, w_center : weights forwarded to ``_score_detection``.
    w_continuity : weight for the spatial-continuity term.
    w_track_stickiness : bonus for same ByteTrack ``track_id``.
    w_of_agreement : weight for OF trace agreement (when of_trace is provided).
    w_velocity : weight for velocity-consistency term.
    vel_history_len : number of recent velocities to maintain.
    of_tight_radius_px : normalisation radius for distance→score conversion.
    of_synthetic_confidence : confidence assigned to synthetic OF candidates.
    frame_dir : directory containing extracted frames (needed for OF).
    seg_frames : segmentation frame list (maps frame_id → frame_file).
    optical_flow_method : ``"auto"`` | ``"sparse"`` | ``"dense"`` | ``"none"``.
    flow_min_keypoints : sparse→dense fallback threshold.
    of_trace : optional dict mapping frame_id → (of_cx, of_cy) from OF trace.

    Returns
    -------
    dict mapping ``frame_id`` → chosen observation.
    """
    use_flow = (
        optical_flow_method != "none"
        and _HAS_NUMPY
        and frame_dir is not None
        and seg_frames is not None
    )

    # Small LRU cache for frames to avoid re-reads
    _frame_cache: dict[int, "np.ndarray | None"] = {}

    def _get_frame(fid: int) -> "np.ndarray | None":
        if fid not in _frame_cache:
            if seg_frames is not None and frame_dir is not None:
                fpath = frame_dir / seg_frames[fid]["frame_file"]
                img = cv2.imread(str(fpath))
                _frame_cache[fid] = img
            else:
                _frame_cache[fid] = None
            # Keep cache bounded
            if len(_frame_cache) > 4:
                oldest = next(iter(_frame_cache))
                del _frame_cache[oldest]
        return _frame_cache.get(fid)

    sorted_fids = sorted(frame_candidates)

    # Helper: score one direction (forward or backward)
    def _score_pass(
        fid_order: list[int],
    ) -> dict[int, dict]:
        """Run a single-direction pass, returning {fid: best_det}."""
        pass_result: dict[int, dict] = {}
        p_bbox: list[int] | None = None
        p_fid: int | None = None
        p_track_id: int | None = None
        vel_history: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=vel_history_len,
        )

        for fid in fid_order:
            candidates = frame_candidates[fid]

            # Filter candidates against OF trace: reject detections far
            # from the OF prediction (likely wrong person).
            if of_trace is not None and fid in of_trace:
                of_cx, of_cy = of_trace[fid]
                near_of = []
                for det in candidates:
                    det_cx = (det["bbox"][0] + det["bbox"][2]) / 2
                    det_cy = (det["bbox"][1] + det["bbox"][3]) / 2
                    dist_to_of = math.sqrt(
                        (det_cx - of_cx) ** 2 + (det_cy - of_cy) ** 2
                    )
                    if dist_to_of < of_tight_radius_px * 2.0:
                        near_of.append(det)
                # Only replace if we still have at least 1 candidate
                if near_of:
                    candidates = near_of
                else:
                    # No real detection within 2×radius of OF — inject a
                    # synthetic OF candidate so the tracker doesn't default
                    # to a distant bystander.
                    if p_bbox is not None and of_synthetic_confidence > 0:
                        p_w = p_bbox[2] - p_bbox[0]
                        p_h = p_bbox[3] - p_bbox[1]
                        syn_x1 = int(round(of_cx - p_w / 2))
                        syn_y1 = int(round(of_cy - p_h / 2))
                        syn_x2 = syn_x1 + p_w
                        syn_y2 = syn_y1 + p_h
                        synthetic = {
                            "bbox": [syn_x1, syn_y1, syn_x2, syn_y2],
                            "confidence": of_synthetic_confidence,
                            "area": p_w * p_h,
                            "track_id": None,
                            "of_synthetic": True,
                        }
                        candidates = candidates + [synthetic]

            if len(candidates) == 1:
                pass_result[fid] = candidates[0]
                # Update velocity history from single candidate
                if p_bbox is not None:
                    det_cx = (candidates[0]["bbox"][0] + candidates[0]["bbox"][2]) / 2
                    det_cy = (candidates[0]["bbox"][1] + candidates[0]["bbox"][3]) / 2
                    prev_cx = (p_bbox[0] + p_bbox[2]) / 2
                    prev_cy = (p_bbox[1] + p_bbox[3]) / 2
                    vel_history.append((det_cx - prev_cx, det_cy - prev_cy))
                p_bbox = candidates[0]["bbox"]
                p_track_id = candidates[0].get("track_id")
                p_fid = fid
            elif len(candidates) > 1:
                # OF-predicted center from previous chosen bbox
                predicted_cx, predicted_cy = None, None
                if p_bbox is not None:
                    pred_cx = (p_bbox[0] + p_bbox[2]) / 2
                    pred_cy = (p_bbox[1] + p_bbox[3]) / 2

                    if use_flow and p_fid is not None:
                        img_prev = _get_frame(p_fid)
                        img_cur = _get_frame(fid)
                        if img_prev is not None and img_cur is not None:
                            dx, dy, _m = _compute_flow_displacement(
                                img_prev, img_cur, p_bbox,
                                method=optical_flow_method,
                                min_keypoints=flow_min_keypoints,
                            )
                            pred_cx += dx
                            pred_cy += dy

                    predicted_cx, predicted_cy = pred_cx, pred_cy

                # OF trace prediction for this frame
                of_pred_cx, of_pred_cy = None, None
                if of_trace is not None and fid in of_trace:
                    of_pred_cx, of_pred_cy = of_trace[fid]

                best_obs = None
                best_score = -1.0

                # Measure spread to detect genuine conflicts
                cxs = [(d["bbox"][0] + d["bbox"][2]) / 2 for d in candidates]
                cys = [(d["bbox"][1] + d["bbox"][3]) / 2 for d in candidates]
                spread = max(max(cxs) - min(cxs), max(cys) - min(cys))
                is_conflict = spread > 100.0

                for det in candidates:
                    quality = _score_detection(
                        det, cx, cy, max_dist, w_conf, w_center,
                    )

                    # Continuity bonus
                    if predicted_cx is not None and predicted_cy is not None:
                        det_cx = (det["bbox"][0] + det["bbox"][2]) / 2
                        det_cy = (det["bbox"][1] + det["bbox"][3]) / 2
                        dist = math.sqrt(
                            (det_cx - predicted_cx) ** 2
                            + (det_cy - predicted_cy) ** 2
                        )
                        continuity = max(0.0, 1.0 - dist / of_tight_radius_px)
                    else:
                        continuity = 0.5

                    # Track-ID stickiness
                    det_track_id = det.get("track_id")
                    stickiness = (
                        1.0
                        if p_track_id is not None
                        and det_track_id is not None
                        and det_track_id == p_track_id
                        else 0.0
                    )

                    # OF agreement
                    of_agreement = 0.0
                    if det.get("of_synthetic"):
                        # Synthetic candidates are placed AT the OF
                        # prediction, so OF agreement is tautologically 1.0.
                        # Zero it out to avoid circular self-reinforcement.
                        of_agreement = 0.0
                    elif of_pred_cx is not None and of_pred_cy is not None:
                        det_cx = (det["bbox"][0] + det["bbox"][2]) / 2
                        det_cy = (det["bbox"][1] + det["bbox"][3]) / 2
                        dist = math.sqrt(
                            (det_cx - of_pred_cx) ** 2
                            + (det_cy - of_pred_cy) ** 2
                        )
                        of_agreement = max(0.0, 1.0 - dist / of_tight_radius_px)

                    # Velocity consistency
                    vel_score = 0.5  # neutral default
                    if p_bbox is not None:
                        det_cx = (det["bbox"][0] + det["bbox"][2]) / 2
                        det_cy = (det["bbox"][1] + det["bbox"][3]) / 2
                        p_cx = (p_bbox[0] + p_bbox[2]) / 2
                        p_cy = (p_bbox[1] + p_bbox[3]) / 2
                        cand_vx = det_cx - p_cx
                        cand_vy = det_cy - p_cy
                        vel_score = _velocity_consistency(
                            vel_history, cand_vx, cand_vy,
                        )

                    # During genuine conflicts: OF dominates, stickiness reduced
                    if is_conflict and of_pred_cx is not None:
                        effective_of_weight = w_of_agreement * 3.0
                        effective_stickiness = w_track_stickiness * 0.2
                    else:
                        effective_of_weight = w_of_agreement
                        effective_stickiness = w_track_stickiness

                    combined = (
                        quality
                        + w_continuity * continuity
                        + effective_stickiness * stickiness
                        + effective_of_weight * of_agreement
                        + w_velocity * vel_score
                    )
                    if combined > best_score:
                        best_score = combined
                        best_obs = det

                pass_result[fid] = best_obs  # type: ignore[assignment]

                chosen = pass_result[fid]
                # Update velocity history from chosen detection
                if p_bbox is not None:
                    chosen_cx = (chosen["bbox"][0] + chosen["bbox"][2]) / 2
                    chosen_cy = (chosen["bbox"][1] + chosen["bbox"][3]) / 2
                    prev_cx = (p_bbox[0] + p_bbox[2]) / 2
                    prev_cy = (p_bbox[1] + p_bbox[3]) / 2
                    vel_history.append(
                        (chosen_cx - prev_cx, chosen_cy - prev_cy),
                    )
                p_bbox = chosen["bbox"]
                p_track_id = chosen.get("track_id")
                p_fid = fid

        return pass_result

    # Forward pass only (bidirectional caused regressions on good videos)
    selected_obs = _score_pass(sorted_fids)

    return selected_obs


# ---------------------------------------------------------------------------
# Gap-filling helpers
# ---------------------------------------------------------------------------

def _fill_gaps(
    frame_bboxes: dict[int, dict],
    n_frames: int,
    img_w: int,
    img_h: int,
) -> None:
    """Fill ALL detection gaps to guarantee one bbox per frame (in-place).

    Internal gaps are linearly interpolated between the two nearest
    detected anchors.  Leading gaps (before the first detection) and
    trailing gaps (after the last detection) are filled by propagating
    the nearest anchor's bbox.
    """
    detected_ids = sorted(frame_bboxes.keys())
    if not detected_ids:
        return

    # --- Leading gap (before first detection) ---
    first = detected_ids[0]
    if first > 0:
        anchor = frame_bboxes[first]
        for fid in range(0, first):
            frame_bboxes[fid] = {
                "bbox": list(anchor["bbox"]),
                "confidence": 0.0,
                "detected": False,
                "interpolated": True,
            }

    # --- Internal gaps ---
    for i in range(len(detected_ids) - 1):
        a_id = detected_ids[i]
        b_id = detected_ids[i + 1]
        gap_len = b_id - a_id - 1
        if gap_len == 0:
            continue

        a_box = frame_bboxes[a_id]["bbox"]
        b_box = frame_bboxes[b_id]["bbox"]

        for fid in range(a_id + 1, b_id):
            t = (fid - a_id) / (b_id - a_id)
            interp_bbox = [
                int(round(a_box[j] + t * (b_box[j] - a_box[j])))
                for j in range(4)
            ]
            # Clamp to image bounds
            interp_bbox[0] = max(0, interp_bbox[0])
            interp_bbox[1] = max(0, interp_bbox[1])
            interp_bbox[2] = min(img_w, interp_bbox[2])
            interp_bbox[3] = min(img_h, interp_bbox[3])

            frame_bboxes[fid] = {
                "bbox": interp_bbox,
                "confidence": 0.0,
                "detected": False,
                "interpolated": True,
            }

    # --- Trailing gap (after last detection) ---
    last = detected_ids[-1]
    if last < n_frames - 1:
        anchor = frame_bboxes[last]
        for fid in range(last + 1, n_frames):
            frame_bboxes[fid] = {
                "bbox": list(anchor["bbox"]),
                "confidence": 0.0,
                "detected": False,
                "interpolated": True,
            }


# ---------------------------------------------------------------------------
# Optical-flow helpers
# ---------------------------------------------------------------------------

# Lucas-Kanade parameters for sparse optical flow
_LK_PARAMS: dict = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Farneback parameters for dense optical flow
_FARNEBACK_PARAMS: dict = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def _compute_camera_motion(
    img_prev: "np.ndarray",
    img_next: "np.ndarray",
    exclude_bbox: list[int] | None = None,
    exclude_margin: float = 1.5,
    method: str = "orb",
    min_features: int = 20,
    ransac_threshold: float = 3.0,
) -> tuple[float, float, float, str]:
    """Estimate global camera motion between two frames.

    Uses feature matching on background regions (excluding the skier ROI)
    to compute a homography representing camera pan, tilt, and rotation.
    This enables separating camera motion from object motion in optical flow.

    Parameters
    ----------
    img_prev : previous frame (grayscale or BGR).
    img_next : next frame (grayscale or BGR).
    exclude_bbox : bounding box ``[x1, y1, x2, y2]`` to exclude from feature
        detection (typically the skier). If None, uses full frame.
    exclude_margin : multiplier for exclusion zone (1.5 = 1.5× bbox area).
    method : ``"orb"`` | ``"ecc"`` | ``"none"``. ORB is fastest and recommended.
    min_features : minimum matched features required for valid homography.
    ransac_threshold : pixel threshold for RANSAC outlier rejection.

    Returns
    -------
    ``(dx, dy, d_theta, method_used)`` — camera translation (pixels) and
    rotation (radians), plus method that succeeded. Returns (0, 0, 0, "none")
    if estimation fails.
    """
    if not _HAS_NUMPY:
        return 0.0, 0.0, 0.0, "none"

    if method == "none":
        return 0.0, 0.0, 0.0, "none"

    # Convert to grayscale if needed
    if len(img_prev.shape) == 3:
        gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    else:
        gray_prev = img_prev
    if len(img_next.shape) == 3:
        gray_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
    else:
        gray_next = img_next

    h, w = gray_prev.shape

    # Create mask to exclude skier region from feature detection
    mask = None
    if exclude_bbox is not None:
        x1, y1, x2, y2 = exclude_bbox
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Expand exclusion zone
        ex_w = int(bw * exclude_margin)
        ex_h = int(bh * exclude_margin)
        ex_x1 = max(0, cx - ex_w // 2)
        ex_y1 = max(0, cy - ex_h // 2)
        ex_x2 = min(w, cx + ex_w // 2)
        ex_y2 = min(h, cy + ex_h // 2)

        # Create mask (255 = search region, 0 = excluded)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask[ex_y1:ex_y2, ex_x1:ex_x2] = 0

    # --- ORB Feature Matching ---
    if method == "orb":
        try:
            # Detect and compute ORB features
            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(gray_prev, mask)
            kp2, des2 = orb.detectAndCompute(gray_next, mask)

            if des1 is None or des2 is None or len(kp1) < min_features or len(kp2) < min_features:
                return 0.0, 0.0, 0.0, "none"

            # Match features using Hamming distance (ORB is binary descriptor)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            if len(matches) < min_features:
                return 0.0, 0.0, 0.0, "none"

            # Extract matched point coordinates
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            # Estimate homography with RANSAC
            H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_threshold)

            if H is None or inliers is None or np.sum(inliers) < min_features:
                return 0.0, 0.0, 0.0, "none"

            # Extract translation from homography
            dx = float(H[0, 2])
            dy = float(H[1, 2])

            # Extract rotation (approximate from rotation component)
            # For small rotations: theta ≈ atan2(H[1,0], H[0,0])
            d_theta = float(np.arctan2(H[1, 0], H[0, 0]))

            return dx, dy, d_theta, "orb"

        except Exception:
            return 0.0, 0.0, 0.0, "none"

    # --- ECC (Enhanced Correlation Coefficient) ---
    elif method == "ecc":
        try:
            # ECC requires good initialization
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)

            # Apply mask by zeroing excluded regions (ECC doesn't support mask directly)
            if mask is not None:
                gray_prev_masked = gray_prev.copy()
                gray_next_masked = gray_next.copy()
                gray_prev_masked[mask == 0] = 0
                gray_next_masked[mask == 0] = 0
            else:
                gray_prev_masked = gray_prev
                gray_next_masked = gray_next

            # Compute affine transform
            _, warp_matrix = cv2.findTransformECC(
                gray_prev_masked, gray_next_masked, warp_matrix,
                cv2.MOTION_EUCLIDEAN, criteria
            )

            dx = float(warp_matrix[0, 2])
            dy = float(warp_matrix[1, 2])
            d_theta = float(np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0]))

            return dx, dy, d_theta, "ecc"

        except Exception:
            return 0.0, 0.0, 0.0, "none"

    return 0.0, 0.0, 0.0, "none"


def _compute_flow_displacement(
    img_prev: "np.ndarray",
    img_next: "np.ndarray",
    bbox: list[int],
    method: str = "auto",
    min_keypoints: int = 5,
    camera_motion: tuple[float, float, float] | None = None,
) -> tuple[float, float, str]:
    """Compute bbox displacement between two frames via optical flow.

    Parameters
    ----------
    img_prev : previous frame (BGR).
    img_next : next frame (BGR).
    bbox : reference bounding box ``[x1, y1, x2, y2]`` in ``img_prev``.
    method : ``"auto"`` | ``"sparse"`` | ``"dense"``.
    min_keypoints : sparse→dense fallback threshold (for ``"auto"``).
    camera_motion : optional ``(dx_cam, dy_cam, d_theta_cam)`` to subtract
        from measured flow to isolate object motion.

    Returns
    -------
    ``(dx, dy, method_used)`` — pixel displacement of the bbox centre
    between the two frames (with camera motion compensation if provided)
    and the method that produced the result.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # Pad the ROI for context (×1.5)
    pad_x = max(int(bw * 0.25), 5)
    pad_y = max(int(bh * 0.25), 5)
    h, w = img_prev.shape[:2]
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(w, x2 + pad_x)
    ry2 = min(h, y2 + pad_y)

    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    gray_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)

    try_sparse = method in ("auto", "sparse")
    try_dense = method in ("auto", "dense")
    used = "none"

    # --- Sparse Lucas-Kanade ---
    if try_sparse:
        # Detect features inside the bbox region
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255

        pts = cv2.goodFeaturesToTrack(
            gray_prev, maxCorners=50, qualityLevel=0.05,
            minDistance=max(3, min(bw, bh) // 8), mask=roi_mask,
        )

        if pts is not None and len(pts) >= min_keypoints:
            pts_next, status, _err = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_next, pts, None, **_LK_PARAMS,
            )
            # Keep only successfully tracked points
            good_mask = status.ravel() == 1
            if good_mask.sum() >= max(3, min_keypoints // 2):
                p0 = pts[good_mask].reshape(-1, 2)
                p1 = pts_next[good_mask].reshape(-1, 2)
                displacements = p1 - p0

                # Outlier rejection via median absolute deviation
                med = np.median(displacements, axis=0)
                mad = np.median(np.abs(displacements - med), axis=0)
                mad = np.maximum(mad, 1e-6)  # avoid division by zero
                mask_inlier = np.all(
                    np.abs(displacements - med) < 3.0 * mad, axis=1,
                )
                if mask_inlier.sum() >= 2:
                    dx = float(np.median(displacements[mask_inlier, 0]))
                    dy = float(np.median(displacements[mask_inlier, 1]))

                    # Subtract camera motion if provided
                    if camera_motion is not None:
                        dx_cam, dy_cam, _ = camera_motion
                        dx -= dx_cam
                        dy -= dy_cam

                    return dx, dy, "sparse"

        # Not enough keypoints — try dense if auto
        if method == "sparse":
            return 0.0, 0.0, "none"

    # --- Dense Farneback (on ROI region) ---
    if try_dense:
        roi_prev = gray_prev[ry1:ry2, rx1:rx2]
        roi_next = gray_next[ry1:ry2, rx1:rx2]

        if roi_prev.size == 0 or roi_next.size == 0:
            return 0.0, 0.0, "none"

        flow = cv2.calcOpticalFlowFarneback(
            roi_prev, roi_next, None, **_FARNEBACK_PARAMS,
        )

        # Extract mean flow within the original bbox (mapped to ROI coords)
        bx1 = x1 - rx1
        by1 = y1 - ry1
        bx2 = x2 - rx1
        by2 = y2 - ry1
        bbox_flow = flow[by1:by2, bx1:bx2]

        if bbox_flow.size == 0:
            return 0.0, 0.0, "none"

        dx = float(np.mean(bbox_flow[..., 0]))
        dy = float(np.mean(bbox_flow[..., 1]))

        # Subtract camera motion if provided
        if camera_motion is not None:
            dx_cam, dy_cam, _ = camera_motion
            dx -= dx_cam
            dy -= dy_cam

        used = "dense"
        return dx, dy, used

    return 0.0, 0.0, "none"


def _propagate_bbox(
    bbox: list[int], dx: float, dy: float, img_w: int, img_h: int,
) -> list[int]:
    """Shift a bbox by ``(dx, dy)`` and clamp to image bounds."""
    x1 = int(round(bbox[0] + dx))
    y1 = int(round(bbox[1] + dy))
    x2 = int(round(bbox[2] + dx))
    y2 = int(round(bbox[3] + dy))
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))
    return [x1, y1, x2, y2]


def _read_gray(frame_dir: Path, frame_file: str) -> "np.ndarray | None":
    """Read a frame and return it as BGR (for reuse) or None."""
    img = cv2.imread(str(frame_dir / frame_file))
    return img


def _fill_gaps_optical_flow(
    frame_bboxes: dict[int, dict],
    n_frames: int,
    img_w: int,
    img_h: int,
    frame_dir: Path,
    seg_frames: list[dict],
    *,
    method: str = "auto",
    min_keypoints: int = 5,
    max_extrapolate: int = 30,
    cmc_enabled: bool = True,
    cmc_method: str = "orb",
    cmc_exclude_margin: float = 1.5,
    cmc_min_features: int = 20,
    cmc_ransac_threshold: float = 3.0,
) -> tuple[dict[int, tuple[float, float]], str]:
    """Fill detection gaps using optical flow (in-place).

    For internal gaps, propagates forward from the left anchor and backward
    from the right anchor, then blends the two estimates linearly.  For
    leading/trailing gaps, propagates using flow up to ``max_extrapolate``
    frames, then copies the anchor for the remainder.

    Parameters
    ----------
    frame_bboxes : mapping ``frame_id → {bbox, confidence, detected, interpolated}``.
        Modified in-place.
    n_frames : total number of frames.
    img_w, img_h : image dimensions.
    frame_dir : path to frame images.
    seg_frames : list of segmentation frame dicts (for filenames).
    method : optical flow method.
    min_keypoints : sparse→dense fallback threshold.
    max_extrapolate : maximum frames to propagate for leading/trailing gaps.
    cmc_enabled : enable camera motion compensation if True.
    cmc_method : camera motion method ("orb" | "ecc" | "none").
    cmc_exclude_margin : exclusion margin multiplier for skier bbox.
    cmc_min_features : minimum matches for valid camera motion estimate.
    cmc_ransac_threshold : RANSAC outlier threshold for homography.

    Returns
    -------
    ``(flow_velocities, method_summary)`` — per-frame ``(dx, dy)`` map and
    a string summarising the methods used (``"sparse"``, ``"dense"``, or
    ``"mixed"``).
    """
    detected_ids = sorted(frame_bboxes.keys())
    if not detected_ids:
        return {}, "none"

    flow_velocities: dict[int, tuple[float, float]] = {}
    methods_used: set[str] = set()

    # Cache for loaded frame images (keep limited sliding window)
    _img_cache: dict[int, "np.ndarray"] = {}

    def _get_frame_img(fid: int) -> "np.ndarray | None":
        if fid in _img_cache:
            return _img_cache[fid]
        if 0 <= fid < n_frames:
            img = _read_gray(frame_dir, seg_frames[fid]["frame_file"])
            # Keep cache bounded to ~20 frames
            if len(_img_cache) > 20:
                oldest = min(_img_cache.keys())
                del _img_cache[oldest]
            if img is not None:
                _img_cache[fid] = img
            return img
        return None

    # --- Camera motion cache (computed on-demand per frame pair) ---
    camera_motion: dict[int, tuple[float, float, float]] = {}

    def _get_camera_motion(
        fid_prev: int,
        fid_next: int,
        exclude_bbox: list[int] | None,
    ) -> tuple[float, float, float] | None:
        if not cmc_enabled or cmc_method == "none":
            return None
        if fid_prev in camera_motion:
            return camera_motion[fid_prev]
        img_prev = _get_frame_img(fid_prev)
        img_next = _get_frame_img(fid_next)
        if img_prev is None or img_next is None:
            return None
        dx_cam, dy_cam, d_theta, _ = _compute_camera_motion(
            img_prev, img_next,
            exclude_bbox=exclude_bbox,
            exclude_margin=cmc_exclude_margin,
            method=cmc_method,
            min_features=cmc_min_features,
            ransac_threshold=cmc_ransac_threshold,
        )
        camera_motion[fid_prev] = (dx_cam, dy_cam, d_theta)
        return camera_motion[fid_prev]

    # --- Leading gap (before first detection) ---
    first = detected_ids[0]
    if first > 0:
        anchor_bbox = list(frame_bboxes[first]["bbox"])
        n_lead = min(first, max_extrapolate)
        # Propagate backward from anchor
        cur_bbox = list(anchor_bbox)
        for step in range(1, n_lead + 1):
            fid = first - step
            img_a = _get_frame_img(first - step + 1)
            img_b = _get_frame_img(fid)
            if img_a is not None and img_b is not None:
                dx, dy, m = _compute_flow_displacement(
                    img_a, img_b, cur_bbox, method=method,
                    min_keypoints=min_keypoints,
                    camera_motion=_get_camera_motion(first - step + 1, fid, cur_bbox),
                )
                if m != "none":
                    methods_used.add(m)
                cur_bbox = _propagate_bbox(cur_bbox, dx, dy, img_w, img_h)
                flow_velocities[fid] = (dx, dy)
            frame_bboxes[fid] = {
                "bbox": list(cur_bbox),
                "confidence": 0.0,
                "detected": False,
                "interpolated": True,
            }
        # Remaining leading frames beyond max_extrapolate: copy farthest
        if first > max_extrapolate:
            farthest = frame_bboxes[first - n_lead]["bbox"]
            for fid in range(0, first - n_lead):
                frame_bboxes[fid] = {
                    "bbox": list(farthest),
                    "confidence": 0.0,
                    "detected": False,
                    "interpolated": True,
                }

    # --- Internal gaps ---
    for i in range(len(detected_ids) - 1):
        a_id = detected_ids[i]
        b_id = detected_ids[i + 1]
        gap_len = b_id - a_id - 1
        if gap_len == 0:
            continue

        a_box = frame_bboxes[a_id]["bbox"]
        b_box = frame_bboxes[b_id]["bbox"]

        # Forward propagation from anchor a
        fwd_bboxes: dict[int, list[int]] = {}
        fwd_vel: dict[int, tuple[float, float]] = {}
        cur = list(a_box)
        for fid in range(a_id + 1, b_id):
            img_prev = _get_frame_img(fid - 1)
            img_cur = _get_frame_img(fid)
            if img_prev is not None and img_cur is not None:
                dx, dy, m = _compute_flow_displacement(
                    img_prev, img_cur, cur, method=method,
                    min_keypoints=min_keypoints,
                    camera_motion=_get_camera_motion(fid - 1, fid, cur),
                )
                if m != "none":
                    methods_used.add(m)
                cur = _propagate_bbox(cur, dx, dy, img_w, img_h)
                fwd_vel[fid] = (dx, dy)
            fwd_bboxes[fid] = list(cur)

        # Backward propagation from anchor b
        bwd_bboxes: dict[int, list[int]] = {}
        bwd_vel: dict[int, tuple[float, float]] = {}
        cur = list(b_box)
        for fid in range(b_id - 1, a_id, -1):
            img_next = _get_frame_img(fid + 1)
            img_cur = _get_frame_img(fid)
            if img_next is not None and img_cur is not None:
                dx, dy, m = _compute_flow_displacement(
                    img_next, img_cur, cur, method=method,
                    min_keypoints=min_keypoints,
                    camera_motion=_get_camera_motion(fid + 1, fid, cur),
                )
                if m != "none":
                    methods_used.add(m)
                cur = _propagate_bbox(cur, dx, dy, img_w, img_h)
                bwd_vel[fid] = (dx, dy)
            bwd_bboxes[fid] = list(cur)

        # Blend forward and backward estimates
        for fid in range(a_id + 1, b_id):
            t = (fid - a_id) / (b_id - a_id)  # 0→1 as we go a→b
            fwd = fwd_bboxes.get(fid, list(a_box))
            bwd = bwd_bboxes.get(fid, list(b_box))
            blended = [
                int(round((1.0 - t) * fwd[j] + t * bwd[j]))
                for j in range(4)
            ]
            # Clamp
            blended[0] = max(0, min(blended[0], img_w - 1))
            blended[1] = max(0, min(blended[1], img_h - 1))
            blended[2] = max(blended[0] + 1, min(blended[2], img_w))
            blended[3] = max(blended[1] + 1, min(blended[3], img_h))

            frame_bboxes[fid] = {
                "bbox": blended,
                "confidence": 0.0,
                "detected": False,
                "interpolated": True,
            }
            # Store blended velocity
            fv = fwd_vel.get(fid, (0.0, 0.0))
            bv = bwd_vel.get(fid, (0.0, 0.0))
            flow_velocities[fid] = (
                (1.0 - t) * fv[0] + t * bv[0],
                (1.0 - t) * fv[1] + t * bv[1],
            )

    # --- Trailing gap (after last detection) ---
    last = detected_ids[-1]
    if last < n_frames - 1:
        anchor_bbox = list(frame_bboxes[last]["bbox"])
        n_trail = min(n_frames - 1 - last, max_extrapolate)
        cur_bbox = list(anchor_bbox)
        for step in range(1, n_trail + 1):
            fid = last + step
            img_a = _get_frame_img(fid - 1)
            img_b = _get_frame_img(fid)
            if img_a is not None and img_b is not None:
                dx, dy, m = _compute_flow_displacement(
                    img_a, img_b, cur_bbox, method=method,
                    min_keypoints=min_keypoints,
                    camera_motion=_get_camera_motion(fid - 1, fid, cur_bbox),
                )
                if m != "none":
                    methods_used.add(m)
                cur_bbox = _propagate_bbox(cur_bbox, dx, dy, img_w, img_h)
                flow_velocities[fid] = (dx, dy)
            frame_bboxes[fid] = {
                "bbox": list(cur_bbox),
                "confidence": 0.0,
                "detected": False,
                "interpolated": True,
            }
        # Copy farthest for remainder
        if n_frames - 1 - last > max_extrapolate:
            farthest = frame_bboxes[last + n_trail]["bbox"]
            for fid in range(last + n_trail + 1, n_frames):
                frame_bboxes[fid] = {
                    "bbox": list(farthest),
                    "confidence": 0.0,
                    "detected": False,
                    "interpolated": True,
                }

    # Summarise methods
    if not methods_used:
        method_summary = "none"
    elif len(methods_used) == 1:
        method_summary = methods_used.pop()
    else:
        method_summary = "mixed"

    return flow_velocities, method_summary


# ---------------------------------------------------------------------------
# Identity guard optical flow trace
# ---------------------------------------------------------------------------

def _maintain_of_trace(
    selected_obs: dict[int, dict],
    n_frames: int,
    img_w: int,
    img_h: int,
    frame_dir: Path,
    seg_frames: list[dict],
    *,
    method: str = "auto",
    min_keypoints: int = 5,
    reanchor_interval: int = 50,
    reanchor_min_conf: float = 0.5,
    max_drift_px: float = 200.0,
) -> dict[int, tuple[float, float]]:
    """Maintain a continuous optical-flow trace to guard against identity switches.

    Runs a single-point Lucas-Kanade tracker through the entire video, seeded
    at the first detected bbox centre. Periodically re-anchors to confident
    detections to prevent drift accumulation. The OF trace is independent of
    the detection-based tracking and serves as a reference for identity validation.

    Parameters
    ----------
    selected_obs : dict mapping frame_id → detection observation (with bbox, confidence).
    n_frames : total number of frames in the video.
    img_w, img_h : image dimensions.
    frame_dir : directory containing frame images.
    seg_frames : list of segmentation frame dicts (for filenames).
    method : optical flow method ("auto" | "sparse" | "dense").
    min_keypoints : sparse→dense fallback threshold.
    reanchor_interval : re-anchor to a detection every N detected frames.
    reanchor_min_conf : minimum confidence to use a detection as re-anchor.
    max_drift_px : force re-anchor if cumulative drift since last anchor exceeds this.

    Returns
    -------
    Dict mapping frame_id → (of_cx, of_cy) — the OF-predicted skier centre for
    every frame in the video.
    """
    if not selected_obs or not _HAS_NUMPY:
        return {}

    # Find first detected frame to seed the OF trace
    detected_ids = sorted(selected_obs.keys())
    if not detected_ids:
        return {}

    first_fid = detected_ids[0]
    first_bbox = selected_obs[first_fid]["bbox"]
    seed_cx = (first_bbox[0] + first_bbox[2]) / 2.0
    seed_cy = (first_bbox[1] + first_bbox[3]) / 2.0

    of_trace: dict[int, tuple[float, float]] = {first_fid: (seed_cx, seed_cy)}

    # Cache frames to avoid repeated reads
    _frame_cache: dict[int, "np.ndarray | None"] = {}

    def _get_frame_img(fid: int) -> "np.ndarray | None":
        if fid in _frame_cache:
            return _frame_cache[fid]
        if 0 <= fid < n_frames:
            img = _read_gray(frame_dir, seg_frames[fid]["frame_file"])
            if len(_frame_cache) > 30:
                oldest = min(_frame_cache.keys())
                del _frame_cache[oldest]
            if img is not None:
                _frame_cache[fid] = img
            return img
        return None

    # Track state for re-anchoring
    current_cx, current_cy = seed_cx, seed_cy
    anchor_cx, anchor_cy = seed_cx, seed_cy
    cumulative_drift = 0.0
    detected_since_anchor = 0

    # Forward trace from first detected frame
    for fid in range(first_fid + 1, n_frames):
        # Try to compute OF from previous frame
        img_prev = _get_frame_img(fid - 1)
        img_cur = _get_frame_img(fid)

        dx, dy = 0.0, 0.0
        if img_prev is not None and img_cur is not None:
            # Create a temporary bbox around the current predicted position
            pad = 30
            temp_bbox = [
                max(0, int(current_cx - pad)),
                max(0, int(current_cy - pad)),
                min(img_w, int(current_cx + pad)),
                min(img_h, int(current_cy + pad)),
            ]
            dx, dy, _ = _compute_flow_displacement(
                img_prev, img_cur, temp_bbox,
                method=method, min_keypoints=min_keypoints,
            )

        # Update predicted position
        current_cx += dx
        current_cy += dy

        # Clamp to image bounds
        current_cx = max(0, min(current_cx, img_w - 1))
        current_cy = max(0, min(current_cy, img_h - 1))

        of_trace[fid] = (current_cx, current_cy)

        # Track cumulative drift since last anchor
        cumulative_drift += math.sqrt(dx * dx + dy * dy)

        # Check for re-anchoring opportunity
        if fid in selected_obs:
            det = selected_obs[fid]
            det_conf = det.get("confidence", 0.0)

            if det_conf >= reanchor_min_conf:
                detected_since_anchor += 1

                # Re-anchor periodically or if drift is too large
                should_reanchor = (
                    detected_since_anchor >= reanchor_interval
                    or cumulative_drift > max_drift_px
                )

                if should_reanchor:
                    det_bbox = det["bbox"]
                    anchor_cx = (det_bbox[0] + det_bbox[2]) / 2.0
                    anchor_cy = (det_bbox[1] + det_bbox[3]) / 2.0
                    current_cx = anchor_cx
                    current_cy = anchor_cy

                    of_trace[fid] = (anchor_cx, anchor_cy)
                    cumulative_drift = 0.0
                    detected_since_anchor = 0

    # Backward trace from first detected frame (fill leading frames)
    if first_fid > 0:
        current_cx, current_cy = seed_cx, seed_cy

        for fid in range(first_fid - 1, -1, -1):
            img_cur = _get_frame_img(fid)
            img_next = _get_frame_img(fid + 1)

            dx, dy = 0.0, 0.0
            if img_cur is not None and img_next is not None:
                pad = 30
                temp_bbox = [
                    max(0, int(current_cx - pad)),
                    max(0, int(current_cy - pad)),
                    min(img_w, int(current_cx + pad)),
                    min(img_h, int(current_cy + pad)),
                ]
                dx, dy, _ = _compute_flow_displacement(
                    img_next, img_cur, temp_bbox,
                    method=method, min_keypoints=min_keypoints,
                )

            current_cx += dx
            current_cy += dy
            current_cx = max(0, min(current_cx, img_w - 1))
            current_cy = max(0, min(current_cy, img_h - 1))

            of_trace[fid] = (current_cx, current_cy)

    return of_trace


def _build_multi_candidate_of_trace(
    frame_candidates: dict[int, list[dict]],
    n_frames: int,
    img_w: int,
    img_h: int,
    frame_dir: Path,
    seg_frames: list[dict],
    *,
    seed_obs: dict | None = None,
    method: str = "auto",
    min_keypoints: int = 5,
    max_reanchor_dist_px: float = 150.0,
    reanchor_interval: int = 5,
    max_drift_px: float = 150.0,
) -> dict[int, tuple[float, float]]:
    """Build an OF trace that can re-anchor to ANY candidate detection.

    Unlike ``_maintain_of_trace`` which only re-anchors to detections from
    a single pre-selected track, this function examines ALL candidate
    detections at each frame and re-anchors to the closest one within
    ``max_reanchor_dist_px``.  This lets the OF trace follow the skier
    seamlessly across ByteTrack track-ID boundaries.

    Parameters
    ----------
    frame_candidates : mapping from ``frame_id`` to list of candidate dicts
        (each with ``bbox``, ``confidence``).
    n_frames : total number of frames in the video.
    img_w, img_h : image dimensions.
    frame_dir : directory containing frame images.
    seg_frames : list of segmentation frame dicts (for filenames).
    seed_obs : optional initial observation to seed from.  If *None*, seeds
        from the highest-confidence detection in the first frame with
        candidates.
    method : optical flow method.
    min_keypoints : sparse→dense fallback threshold.
    max_reanchor_dist_px : only re-anchor to a candidate within this distance
        from the current OF prediction.
    reanchor_interval : re-anchor every N frames that have a close candidate.
    max_drift_px : force re-anchor if cumulative drift exceeds this.

    Returns
    -------
    Dict mapping frame_id → (of_cx, of_cy).
    """
    if not frame_candidates or not _HAS_NUMPY:
        return {}

    # --- Determine seed position ---
    if seed_obs is not None:
        seed_bbox = seed_obs["bbox"]
        seed_fid = seed_obs["frame_id"]
        seed_cx = (seed_bbox[0] + seed_bbox[2]) / 2.0
        seed_cy = (seed_bbox[1] + seed_bbox[3]) / 2.0
    else:
        # Find the highest-confidence detection across all early candidates
        best_seed: dict | None = None
        best_conf = -1.0
        for fid in sorted(frame_candidates):
            for det in frame_candidates[fid]:
                if det.get("confidence", 0) > best_conf:
                    best_conf = det["confidence"]
                    best_seed = det
                    seed_fid = fid
            # Only look at first few frames with candidates
            if best_seed is not None and fid > sorted(frame_candidates)[0] + 50:
                break
        if best_seed is None:
            return {}
        seed_bbox = best_seed["bbox"]
        seed_cx = (seed_bbox[0] + seed_bbox[2]) / 2.0
        seed_cy = (seed_bbox[1] + seed_bbox[3]) / 2.0

    of_trace: dict[int, tuple[float, float]] = {seed_fid: (seed_cx, seed_cy)}

    # Frame cache
    _frame_cache: dict[int, "np.ndarray | None"] = {}

    def _get_frame_img(fid: int) -> "np.ndarray | None":
        if fid in _frame_cache:
            return _frame_cache[fid]
        if 0 <= fid < n_frames:
            img = _read_gray(frame_dir, seg_frames[fid]["frame_file"])
            if len(_frame_cache) > 30:
                oldest = min(_frame_cache.keys())
                del _frame_cache[oldest]
            if img is not None:
                _frame_cache[fid] = img
            return img
        return None

    def _find_closest_candidate(
        fid: int, pred_cx: float, pred_cy: float,
    ) -> tuple[float, float, float] | None:
        """Find the candidate detection closest to (pred_cx, pred_cy).

        Returns (cx, cy, dist) or None if no candidates in this frame.
        """
        if fid not in frame_candidates:
            return None
        best_cx, best_cy, best_dist = 0.0, 0.0, float("inf")
        for det in frame_candidates[fid]:
            bbox = det["bbox"]
            dcx = (bbox[0] + bbox[2]) / 2.0
            dcy = (bbox[1] + bbox[3]) / 2.0
            d = math.sqrt((dcx - pred_cx) ** 2 + (dcy - pred_cy) ** 2)
            if d < best_dist:
                best_dist = d
                best_cx, best_cy = dcx, dcy
        return (best_cx, best_cy, best_dist)

    # --- Forward trace ---
    current_cx, current_cy = seed_cx, seed_cy
    cumulative_drift = 0.0
    frames_since_anchor = 0

    for fid in range(seed_fid + 1, n_frames):
        img_prev = _get_frame_img(fid - 1)
        img_cur = _get_frame_img(fid)

        dx, dy = 0.0, 0.0
        if img_prev is not None and img_cur is not None:
            pad = 30
            temp_bbox = [
                max(0, int(current_cx - pad)),
                max(0, int(current_cy - pad)),
                min(img_w, int(current_cx + pad)),
                min(img_h, int(current_cy + pad)),
            ]
            dx, dy, _ = _compute_flow_displacement(
                img_prev, img_cur, temp_bbox,
                method=method, min_keypoints=min_keypoints,
            )

        current_cx += dx
        current_cy += dy
        current_cx = max(0, min(current_cx, img_w - 1))
        current_cy = max(0, min(current_cy, img_h - 1))

        of_trace[fid] = (current_cx, current_cy)
        cumulative_drift += math.sqrt(dx * dx + dy * dy)
        frames_since_anchor += 1

        # Try to re-anchor to closest candidate
        closest = _find_closest_candidate(fid, current_cx, current_cy)
        if closest is not None:
            c_cx, c_cy, c_dist = closest
            should_reanchor = (
                c_dist < max_reanchor_dist_px
                and (
                    frames_since_anchor >= reanchor_interval
                    or cumulative_drift > max_drift_px
                )
            )
            if should_reanchor:
                current_cx = c_cx
                current_cy = c_cy
                of_trace[fid] = (c_cx, c_cy)
                cumulative_drift = 0.0
                frames_since_anchor = 0

    # --- Backward trace (leading frames) ---
    if seed_fid > 0:
        current_cx, current_cy = seed_cx, seed_cy
        for fid in range(seed_fid - 1, -1, -1):
            img_cur = _get_frame_img(fid)
            img_next = _get_frame_img(fid + 1)

            dx, dy = 0.0, 0.0
            if img_cur is not None and img_next is not None:
                pad = 30
                temp_bbox = [
                    max(0, int(current_cx - pad)),
                    max(0, int(current_cy - pad)),
                    min(img_w, int(current_cx + pad)),
                    min(img_h, int(current_cy + pad)),
                ]
                dx, dy, _ = _compute_flow_displacement(
                    img_next, img_cur, temp_bbox,
                    method=method, min_keypoints=min_keypoints,
                )

            current_cx += dx
            current_cy += dy
            current_cx = max(0, min(current_cx, img_w - 1))
            current_cy = max(0, min(current_cy, img_h - 1))
            of_trace[fid] = (current_cx, current_cy)

    return of_trace


def _validate_identity(
    selected_obs: dict[int, dict],
    of_trace: dict[int, tuple[float, float]],
    *,
    max_jump_px: float = 150.0,
) -> tuple[dict[int, dict], dict[str, object]]:
    """Validate detections against OF trace to reject identity switches (in-place).

    Compares each detection's center against the OF-predicted position. If a
    detection is far from the OF prediction AND far from the previous detection,
    it is likely an identity switch and gets rejected (removed from selected_obs).

    Also detects sequences of wrong-identity detections and rejects them as a
    group until a detection is again close to the OF trace.

    Parameters
    ----------
    selected_obs : dict mapping frame_id → detection (will be modified in-place).
    of_trace : dict mapping frame_id → (of_cx, of_cy) from optical flow trace.
    max_jump_px : threshold for rejecting a detection (in pixels).

    Returns
    -------
    Tuple of (updated selected_obs, metadata_dict) where metadata_dict contains:
        - "rejections": count of rejected detections
        - "jump_stats": summary stats for distances used by the guard
    """
    def _distance_stats(values: list[float]) -> dict[str, float | int] | None:
        if not values:
            return None
        sorted_vals = sorted(values)

        def _quantile(q: float) -> float:
            pos = (len(sorted_vals) - 1) * q
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return sorted_vals[lo]
            weight = pos - lo
            return sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight

        return {
            "count": len(sorted_vals),
            "mean": round(sum(sorted_vals) / len(sorted_vals), 2),
            "p50": round(_quantile(0.5), 2),
            "p90": round(_quantile(0.9), 2),
            "p95": round(_quantile(0.95), 2),
            "max": round(sorted_vals[-1], 2),
        }

    if not of_trace:
        return selected_obs, {"rejections": 0, "jump_stats": None}

    sorted_fids = sorted(selected_obs.keys())
    if not sorted_fids:
        return selected_obs, {"rejections": 0, "jump_stats": None}

    rejections = 0
    in_wrong_identity_sequence = False
    prev_det_cx, prev_det_cy = None, None

    fids_to_reject: set[int] = set()
    of_distances: list[float] = []
    prev_det_distances: list[float] = []
    rejected_of_distances: list[float] = []
    rejected_prev_det_distances: list[float] = []

    for fid in sorted_fids:
        if fid not in of_trace:
            continue

        det = selected_obs[fid]
        det_bbox = det["bbox"]
        det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
        det_cy = (det_bbox[1] + det_bbox[3]) / 2.0

        of_cx, of_cy = of_trace[fid]

        # Distance from detection to OF prediction
        of_dist = math.sqrt((det_cx - of_cx) ** 2 + (det_cy - of_cy) ** 2)
        of_distances.append(of_dist)

        # Distance from detection to previous detection
        prev_det_dist = float('inf')
        if prev_det_cx is not None:
            prev_det_dist = math.sqrt(
                (det_cx - prev_det_cx) ** 2 + (det_cy - prev_det_cy) ** 2
            )
            prev_det_distances.append(prev_det_dist)

        # Check if this looks like an identity switch:
        # Large jump from OF AND large jump from previous detection
        likely_switch = (
            of_dist > max_jump_px
            and prev_det_dist > max_jump_px * 0.5
        )

        if likely_switch:
            if not in_wrong_identity_sequence:
                # Start of a new wrong-identity sequence
                in_wrong_identity_sequence = True

            fids_to_reject.add(fid)
            rejections += 1
            rejected_of_distances.append(of_dist)
            if math.isfinite(prev_det_dist):
                rejected_prev_det_distances.append(prev_det_dist)
        else:
            # Back to a good detection (close to OF or first frame)
            if in_wrong_identity_sequence:
                in_wrong_identity_sequence = False

            prev_det_cx = det_cx
            prev_det_cy = det_cy

    # Remove rejected detections from selected_obs
    for fid in fids_to_reject:
        del selected_obs[fid]

    metadata = {
        "rejections": rejections,
        "jump_stats": {
            "max_jump_px": round(max_jump_px, 2),
            "prev_jump_threshold_px": round(max_jump_px * 0.5, 2),
            "of_dist_px": _distance_stats(of_distances),
            "prev_det_dist_px": _distance_stats(prev_det_distances),
            "rejected_of_dist_px": _distance_stats(rejected_of_distances),
            "rejected_prev_det_dist_px": _distance_stats(
                rejected_prev_det_distances,
            ),
            "over_max_jump_count": sum(1 for d in of_distances if d > max_jump_px),
            "over_prev_jump_count": sum(
                1 for d in prev_det_distances if d > max_jump_px * 0.5
            ),
        },
    }

    return selected_obs, metadata


# ---------------------------------------------------------------------------
# Velocity-aware temporal smoothing
# ---------------------------------------------------------------------------

def _smooth_bboxes_velocity_aware(
    frame_bboxes: dict[int, dict],
    n_frames: int,
    window: int,
    flow_velocities: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Smooth bboxes using a Kalman filter with constant acceleration model.

    Uses a proper Kalman filter (forward pass) + Rauch-Tung-Striebel (RTS)
    backward smoother for optimal trajectory estimation.  The constant
    acceleration motion model better captures skiing physics: gravity-driven
    speed changes, jumps, and terrain transitions.

    Optical-flow velocities are injected as velocity measurements alongside
    bounding-box position/size measurements.  Detected frames receive lower
    measurement noise (higher trust) than interpolated frames.

    Parameters
    ----------
    frame_bboxes : per-frame bbox dict (modified in-place).
    n_frames : total number of frames.
    window : smoothing window; maps to noise scaling.
    flow_velocities : per-frame ``(dx, dy)`` from optical flow.

    Returns
    -------
    Dict mapping frame_id → (vx, vy) smoothed velocity in px/frame.
    """
    import numpy as np

    sorted_ids = sorted(frame_bboxes.keys())
    if len(sorted_ids) < 2:
        return {}

    dt = 1.0  # normalised frame interval

    # --- State: [cx, cy, vx, vy, ax, ay, w, h] --------------------------
    NS = 8  # state dimension
    NM = 4  # position+size measurement dimension

    # State transition matrix (constant acceleration)
    F = np.eye(NS)
    F[0, 2] = dt                # cx += vx·dt
    F[1, 3] = dt                # cy += vy·dt
    F[0, 4] = 0.5 * dt * dt    # cx += ½·ax·dt²
    F[1, 5] = 0.5 * dt * dt    # cy += ½·ay·dt²
    F[2, 4] = dt                # vx += ax·dt
    F[3, 5] = dt                # vy += ay·dt

    # Measurement matrix: observe [cx, cy, w, h]
    H = np.zeros((NM, NS))
    H[0, 0] = 1.0  # cx
    H[1, 1] = 1.0  # cy
    H[2, 6] = 1.0  # w
    H[3, 7] = 1.0  # h

    # Process noise — tuned via window parameter
    scale = max(1.0, float(window))
    Q = np.diag([
        1.0 * scale,   # cx
        1.0 * scale,   # cy
        4.0 * scale,   # vx
        4.0 * scale,   # vy
        8.0 * scale,   # ax  — acceleration most uncertain
        8.0 * scale,   # ay
        0.5 * scale,   # w   — size changes slowly
        0.5 * scale,   # h
    ])

    # Measurement noise — detected vs interpolated
    r_det_pos = 4.0            # position noise for detected frames
    r_interp_pos = 40.0        # position noise for interpolated frames
    r_size = 8.0               # size noise (always moderate)
    R_det = np.diag([r_det_pos, r_det_pos, r_size, r_size])
    R_interp = np.diag([r_interp_pos, r_interp_pos, r_size * 2, r_size * 2])

    # OF velocity measurement: H_v maps state to [vx, vy]
    H_v = np.zeros((2, NS))
    H_v[0, 2] = 1.0  # vx
    H_v[1, 3] = 1.0  # vy
    R_v = np.diag([10.0, 10.0])  # OF velocity noise

    # --- Initialise state from first frame --------------------------------
    fb0 = frame_bboxes[sorted_ids[0]]["bbox"]
    cx0 = (fb0[0] + fb0[2]) / 2.0
    cy0 = (fb0[1] + fb0[3]) / 2.0
    w0 = float(fb0[2] - fb0[0])
    h0 = float(fb0[3] - fb0[1])
    vx0, vy0 = flow_velocities.get(sorted_ids[0], (0.0, 0.0))

    x = np.array([cx0, cy0, vx0, vy0, 0.0, 0.0, w0, h0])
    P = np.diag([5.0, 5.0, 25.0, 25.0, 100.0, 100.0, 10.0, 10.0])

    I_ns = np.eye(NS)

    # Storage for RTS smoother
    x_fwd: list[np.ndarray] = []
    P_fwd: list[np.ndarray] = []
    x_pred_store: list[np.ndarray] = []
    P_pred_store: list[np.ndarray] = []

    # --- Forward Kalman filter --------------------------------------------
    for i, fid in enumerate(sorted_ids):
        if i == 0:
            x_fwd.append(x.copy())
            P_fwd.append(P.copy())
            x_pred_store.append(x.copy())
            P_pred_store.append(P.copy())
            continue

        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Clamp acceleration to physical limits (~2g ≈ 20 m/s² ≈ ~40px/frame²)
        max_acc = 40.0
        x_pred[4] = float(np.clip(x_pred[4], -max_acc, max_acc))
        x_pred[5] = float(np.clip(x_pred[5], -max_acc, max_acc))

        x_pred_store.append(x_pred.copy())
        P_pred_store.append(P_pred.copy())

        # --- Measurement update (position + size) -------------------------
        raw = frame_bboxes[fid]["bbox"]
        z = np.array([
            (raw[0] + raw[2]) / 2.0,
            (raw[1] + raw[3]) / 2.0,
            float(raw[2] - raw[0]),
            float(raw[3] - raw[1]),
        ])

        is_det = frame_bboxes[fid].get("detected", False)
        R_cur = R_det if is_det else R_interp

        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R_cur
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (I_ns - K @ H) @ P_pred

        # --- Optional velocity update from OF -----------------------------
        vx_of, vy_of = flow_velocities.get(fid, (0.0, 0.0))
        if abs(vx_of) > 0.01 or abs(vy_of) > 0.01:
            z_v = np.array([vx_of, vy_of])
            y_v = z_v - H_v @ x
            S_v = H_v @ P @ H_v.T + R_v
            K_v = P @ H_v.T @ np.linalg.inv(S_v)
            x = x + K_v @ y_v
            P = (I_ns - K_v @ H_v) @ P

        x_fwd.append(x.copy())
        P_fwd.append(P.copy())

    # --- RTS backward smoother --------------------------------------------
    n = len(sorted_ids)
    x_smooth: list[np.ndarray] = [np.zeros(NS)] * n
    x_smooth[-1] = x_fwd[-1].copy()

    for i in range(n - 2, -1, -1):
        P_pred = P_pred_store[i + 1]
        try:
            G = P_fwd[i] @ F.T @ np.linalg.inv(P_pred)
        except np.linalg.LinAlgError:
            G = np.zeros((NS, NS))
        x_smooth[i] = x_fwd[i] + G @ (x_smooth[i + 1] - x_pred_store[i + 1])

    # --- Write smoothed bboxes and extract velocities --------------------
    smoothed_velocities: dict[int, tuple[float, float]] = {}
    for i, fid in enumerate(sorted_ids):
        s = x_smooth[i]
        cx_s, cy_s = s[0], s[1]
        w_s = max(10.0, s[6])
        h_s = max(10.0, s[7])
        frame_bboxes[fid]["bbox"] = [
            int(round(cx_s - w_s / 2.0)),
            int(round(cy_s - h_s / 2.0)),
            int(round(cx_s + w_s / 2.0)),
            int(round(cy_s + h_s / 2.0)),
        ]
        smoothed_velocities[fid] = (float(s[2]), float(s[3]))

    return smoothed_velocities


# ---------------------------------------------------------------------------
# Temporal smoothing (legacy EMA — used when optical flow is disabled)
# ---------------------------------------------------------------------------

def _smooth_bboxes(
    frame_bboxes: dict[int, dict],
    n_frames: int,
    window: int,
) -> None:
    """Apply exponential moving average to bbox coordinates (in-place).

    The smoothing only affects frames that have a bbox assigned
    (detected or interpolated).  The EMA factor ``alpha`` is
    derived from the window size as ``2 / (window + 1)``.
    """
    alpha = 2.0 / (window + 1)

    sorted_ids = sorted(frame_bboxes.keys())
    if len(sorted_ids) < 2:
        return

    # Forward EMA
    ema = list(frame_bboxes[sorted_ids[0]]["bbox"])
    smoothed: dict[int, list[float]] = {sorted_ids[0]: list(ema)}

    for fid in sorted_ids[1:]:
        raw = frame_bboxes[fid]["bbox"]
        ema = [alpha * raw[j] + (1 - alpha) * ema[j] for j in range(4)]
        smoothed[fid] = list(ema)

    # Backward EMA (average forward + backward for zero-phase smoothing)
    ema_bw = list(frame_bboxes[sorted_ids[-1]]["bbox"])
    backward: dict[int, list[float]] = {sorted_ids[-1]: list(ema_bw)}

    for fid in reversed(sorted_ids[:-1]):
        raw = frame_bboxes[fid]["bbox"]
        ema_bw = [alpha * raw[j] + (1 - alpha) * ema_bw[j] for j in range(4)]
        backward[fid] = list(ema_bw)

    # Merge forward + backward
    for fid in sorted_ids:
        avg = [
            int(round((smoothed[fid][j] + backward[fid][j]) / 2))
            for j in range(4)
        ]
        frame_bboxes[fid]["bbox"] = avg


# ---------------------------------------------------------------------------
# Fallback when no track IDs are present
# ---------------------------------------------------------------------------

def _write_passthrough(
    seg_data: dict,
    frame_dir: Path,
    output_dir: Path,
    padding_ratio: float,
    crop_dir: Path,
) -> Path:
    """Write a tracking manifest that mirrors the segmentation selection.

    Used when the segmentation manifest has no track IDs (e.g. it was
    produced by an older version of *yolo_seg* that used ``model()``
    instead of ``model.track()``).
    """
    seg_frames = seg_data.get("frames", [])
    manifest_frames: list[dict] = []

    for seg_fr in tqdm(seg_frames, desc="Passthrough crops"):
        idx = seg_fr["frame_id"]
        frame_file = seg_fr["frame_file"]
        img = cv2.imread(str(frame_dir / frame_file))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        if seg_fr.get("detected", False):
            x1, y1, x2, y2 = seg_fr["bbox"]
            bw, bh = x2 - x1, y2 - y1
            pad_x = int(bw * padding_ratio)
            pad_y = int(bh * padding_ratio)
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(img_w, x2 + pad_x)
            py2 = min(img_h, y2 + pad_y)

            crop = img[py1:py2, px1:px2]
            crop_name = f"crop_{idx:06d}.jpg"
            cv2.imwrite(str(crop_dir / crop_name), crop)

            manifest_frames.append({
                "frame_id": idx,
                "frame_file": frame_file,
                "detected": True,
                "interpolated": False,
                "bbox": [x1, y1, x2, y2],
                "bbox_padded": [px1, py1, px2, py2],
                "confidence": seg_fr.get("confidence", 0.0),
                "track_id": None,
                "crop_file": crop_name,
            })
        else:
            crop_name = f"crop_{idx:06d}.jpg"
            cv2.imwrite(str(crop_dir / crop_name), img)
            manifest_frames.append({
                "frame_id": idx,
                "frame_file": frame_file,
                "detected": False,
                "interpolated": False,
                "bbox": [0, 0, img_w, img_h],
                "bbox_padded": [0, 0, img_w, img_h],
                "confidence": 0.0,
                "track_id": None,
                "crop_file": crop_name,
            })

    manifest = {
        "source_segmentation": str(seg_data.get("model", "unknown")),
        "selected_track_id": None,
        "track_score": 0.0,
        "weights": {},
        "smooth_window": 0,
        "interpolate_gaps": False,
        "max_gap_frames": 0,
        "padding_ratio": padding_ratio,
        "n_frames": len(manifest_frames),
        "frames": manifest_frames,
    }
    manifest_path = output_dir / "tracking.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
