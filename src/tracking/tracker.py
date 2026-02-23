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
    merge_top_n: int = 5,
    merge_score_threshold: float = 0.3,
    optical_flow_method: str = "auto",
    flow_max_extrapolate_frames: int = 30,
    flow_min_keypoints: int = 5,
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
    merge_top_n : max number of tracks to merge (if merge_tracks=True).
    merge_score_threshold : minimum score for track inclusion in merge.
    optical_flow_method : ``"auto"`` (sparse LK, fallback to dense),
        ``"sparse"`` (Lucas-Kanade only), ``"dense"`` (Farneback only),
        or ``"none"`` (disable optical flow, use linear interpolation).
    flow_max_extrapolate_frames : max frames to extrapolate via optical
        flow for leading/trailing gaps (beyond this, copy anchor).
    flow_min_keypoints : minimum tracked keypoints before sparse LK
        falls back to dense Farneback (used when method is ``"auto"``).

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
        print(f"  Multi-track merge: enabled (top {merge_top_n}, threshold {merge_score_threshold:.2f})")

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
        # Select top N tracks that meet the score threshold
        selected_tracks = [
            tid for tid, score, _ in track_scores[:merge_top_n]
            if score >= merge_score_threshold
        ]
        if not selected_tracks:
            # Fallback: use the best track even if below threshold
            selected_tracks = [track_scores[0][0]]

        total_detections = sum(len(tracks[tid]) for tid in selected_tracks)
        print(f"  → Merging {len(selected_tracks)} track(s): {selected_tracks[:5]}{'...' if len(selected_tracks) > 5 else ''}")
        print(f"  → Combined detections: {total_detections}")

        # Merge observations from all selected tracks
        # For frames with multiple detections, pick the best one (highest confidence + center score)
        selected_obs = {}
        for tid in selected_tracks:
            for o in tracks[tid]:
                fid = o["frame_id"]
                if fid not in selected_obs:
                    selected_obs[fid] = o
                else:
                    # Pick the better detection (weighted by confidence and center proximity)
                    existing = selected_obs[fid]
                    existing_score = _score_detection(existing, cx, cy, max_dist, w_conf, w_center)
                    new_score = _score_detection(o, cx, cy, max_dist, w_conf, w_center)
                    if new_score > existing_score:
                        selected_obs[fid] = o

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
    # Phase B — Assemble per-frame bboxes: detected → interpolated → extrapolated
    # ------------------------------------------------------------------
    frame_bboxes: dict[int, dict] = {}  # frame_id → {bbox, confidence, source}

    for fid in range(n_frames):
        if fid in selected_obs:
            o = selected_obs[fid]
            frame_bboxes[fid] = {
                "bbox": o["bbox"],
                "confidence": o["confidence"],
                "detected": True,
                "interpolated": False,
            }

    # ------------------------------------------------------------------
    # Phase B — Fill ALL gaps so every frame has exactly one bbox
    # ------------------------------------------------------------------
    use_flow = (
        optical_flow_method != "none"
        and _HAS_NUMPY
    )
    flow_velocities: dict[int, tuple[float, float]] = {}
    of_method_used = "none"

    if use_flow:
        print(f"  Optical flow: {optical_flow_method} "
              f"(min_kp={flow_min_keypoints}, max_extrap={flow_max_extrapolate_frames})")
        flow_velocities, of_method_used = _fill_gaps_optical_flow(
            frame_bboxes, n_frames, img_w, img_h,
            frame_dir, seg_frames,
            method=optical_flow_method,
            min_keypoints=flow_min_keypoints,
            max_extrapolate=flow_max_extrapolate_frames,
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
    if smooth_window > 0 and len(frame_bboxes) > 1:
        if use_flow and flow_velocities:
            _smooth_bboxes_velocity_aware(
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
        })

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


def _compute_flow_displacement(
    img_prev: "np.ndarray",
    img_next: "np.ndarray",
    bbox: list[int],
    method: str = "auto",
    min_keypoints: int = 5,
) -> tuple[float, float, str]:
    """Compute bbox displacement between two frames via optical flow.

    Parameters
    ----------
    img_prev : previous frame (BGR).
    img_next : next frame (BGR).
    bbox : reference bounding box ``[x1, y1, x2, y2]`` in ``img_prev``.
    method : ``"auto"`` | ``"sparse"`` | ``"dense"``.
    min_keypoints : sparse→dense fallback threshold (for ``"auto"``).

    Returns
    -------
    ``(dx, dy, method_used)`` — pixel displacement of the bbox centre
    between the two frames and the method that produced the result.
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
# Velocity-aware temporal smoothing
# ---------------------------------------------------------------------------

def _smooth_bboxes_velocity_aware(
    frame_bboxes: dict[int, dict],
    n_frames: int,
    window: int,
    flow_velocities: dict[int, tuple[float, float]],
) -> None:
    """Smooth bboxes using a velocity-aware bidirectional filter (in-place).

    Uses per-frame optical-flow velocities as a motion prior.  Detected
    frames are trusted more (low measurement noise) while interpolated
    frames weight the motion prediction more heavily.

    Parameters
    ----------
    frame_bboxes : per-frame bbox dict (modified in-place).
    n_frames : total number of frames.
    window : smoothing window; maps to noise ratio.
    flow_velocities : per-frame ``(dx, dy)`` from optical flow.
    """
    alpha = 2.0 / (window + 1)
    # Detected frames get higher measurement trust
    alpha_det = min(1.0, alpha * 2.0)
    alpha_interp = alpha * 0.5

    sorted_ids = sorted(frame_bboxes.keys())
    if len(sorted_ids) < 2:
        return

    def _get_alpha(fid: int) -> float:
        return alpha_det if frame_bboxes[fid].get("detected", False) else alpha_interp

    # Forward pass — state is [cx, cy, w, h] with velocity prediction
    fb0 = frame_bboxes[sorted_ids[0]]["bbox"]
    state_fwd = [
        (fb0[0] + fb0[2]) / 2.0,  # cx
        (fb0[1] + fb0[3]) / 2.0,  # cy
        float(fb0[2] - fb0[0]),     # w
        float(fb0[3] - fb0[1]),     # h
    ]
    smoothed_fwd: dict[int, list[float]] = {sorted_ids[0]: list(state_fwd)}

    for i in range(1, len(sorted_ids)):
        fid = sorted_ids[i]

        raw = frame_bboxes[fid]["bbox"]
        raw_cx = (raw[0] + raw[2]) / 2.0
        raw_cy = (raw[1] + raw[3]) / 2.0
        raw_w = float(raw[2] - raw[0])
        raw_h = float(raw[3] - raw[1])
        measurement = [raw_cx, raw_cy, raw_w, raw_h]

        # Motion prediction from flow velocity
        vx, vy = flow_velocities.get(fid, (0.0, 0.0))
        prediction = [
            state_fwd[0] + vx,
            state_fwd[1] + vy,
            state_fwd[2],  # width assumed stable
            state_fwd[3],  # height assumed stable
        ]

        a = _get_alpha(fid)
        state_fwd = [
            a * measurement[j] + (1.0 - a) * prediction[j]
            for j in range(4)
        ]
        smoothed_fwd[fid] = list(state_fwd)

    # Backward pass
    fb_last = frame_bboxes[sorted_ids[-1]]["bbox"]
    state_bwd = [
        (fb_last[0] + fb_last[2]) / 2.0,
        (fb_last[1] + fb_last[3]) / 2.0,
        float(fb_last[2] - fb_last[0]),
        float(fb_last[3] - fb_last[1]),
    ]
    smoothed_bwd: dict[int, list[float]] = {sorted_ids[-1]: list(state_bwd)}

    for i in range(len(sorted_ids) - 2, -1, -1):
        fid = sorted_ids[i]
        next_fid = sorted_ids[i + 1]

        raw = frame_bboxes[fid]["bbox"]
        raw_cx = (raw[0] + raw[2]) / 2.0
        raw_cy = (raw[1] + raw[3]) / 2.0
        raw_w = float(raw[2] - raw[0])
        raw_h = float(raw[3] - raw[1])
        measurement = [raw_cx, raw_cy, raw_w, raw_h]

        # Reverse velocity from the next frame's flow
        vx, vy = flow_velocities.get(next_fid, (0.0, 0.0))
        prediction = [
            state_bwd[0] - vx,
            state_bwd[1] - vy,
            state_bwd[2],
            state_bwd[3],
        ]

        a = _get_alpha(fid)
        state_bwd = [
            a * measurement[j] + (1.0 - a) * prediction[j]
            for j in range(4)
        ]
        smoothed_bwd[fid] = list(state_bwd)

    # Merge forward + backward → convert (cx, cy, w, h) back to [x1, y1, x2, y2]
    for fid in sorted_ids:
        fwd = smoothed_fwd[fid]
        bwd = smoothed_bwd[fid]
        cx = (fwd[0] + bwd[0]) / 2.0
        cy = (fwd[1] + bwd[1]) / 2.0
        w = (fwd[2] + bwd[2]) / 2.0
        h = (fwd[3] + bwd[3]) / 2.0
        frame_bboxes[fid]["bbox"] = [
            int(round(cx - w / 2.0)),
            int(round(cy - h / 2.0)),
            int(round(cx + w / 2.0)),
            int(round(cy + h / 2.0)),
        ]


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
