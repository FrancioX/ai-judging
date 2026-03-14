"""CoTracker3-based long-gap fill for Phase B.

Replaces Lucas-Kanade optical flow with CoTracker3's offline joint-trajectory
model for internal gaps >= ``min_gap`` frames.  Falls back transparently if
CoTracker is not installed or if the model fails.

Public API:
    fill_long_gaps_cotracker(frame_bboxes, n_frames, img_w, img_h,
                             frame_dir, seg_frames, *, min_gap, resize_h,
                             device) -> None  (modifies frame_bboxes in-place)
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import torch
    from cotracker.predictor import CoTrackerPredictor
    _HAS_COTRACKER = True
except ImportError:
    _HAS_COTRACKER = False


def load_cotracker(device: str = "cpu") -> "CoTrackerPredictor | None":
    """Load CoTracker3 offline model. Returns None if unavailable."""
    if not _HAS_COTRACKER or not _HAS_NUMPY:
        return None
    try:
        # Force CPU: MPS support for CoTracker ops is incomplete
        model = CoTrackerPredictor(checkpoint=None)
        model = model.eval().to("cpu")
        return model
    except Exception as exc:
        print(f"  ⚠ CoTracker load failed: {exc}")
        return None


def fill_long_gaps_cotracker(
    frame_bboxes: dict[int, dict],
    img_w: int,
    img_h: int,
    frame_dir: Path,
    seg_frames: list[dict],
    model: "CoTrackerPredictor",
    *,
    min_gap: int = 20,
    resize_h: int = 320,
    min_visible_score: float = 0.5,
) -> None:
    """Replace LK fill with CoTracker3 for detected gaps >= min_gap (in-place).

    Parameters
    ----------
    frame_bboxes : mapping frame_id → {bbox, detected, interpolated, ...}
        Modified in-place for gap frames in qualifying long gaps.
    img_w, img_h : original image dimensions (pixels).
    frame_dir : directory of extracted frame images.
    seg_frames : segmentation frame list (frame_id → frame_file).
    model : loaded CoTrackerPredictor instance.
    min_gap : only re-fill gaps this length or longer.
    resize_h : resize frames to this height before CoTracker inference
        (width scaled proportionally; smaller = faster / less memory).
    """
    if not _HAS_COTRACKER or not _HAS_NUMPY or model is None:
        return

    # Identify all detected (non-interpolated) frame IDs in order
    detected_ids = sorted(
        fid for fid, fb in frame_bboxes.items()
        if fb.get("detected") and not fb.get("of_synthetic")
    )
    if len(detected_ids) < 2:
        return

    replaced = 0
    failed = 0

    for i in range(len(detected_ids) - 1):
        a_id = detected_ids[i]
        b_id = detected_ids[i + 1]
        gap_len = b_id - a_id - 1
        if gap_len < min_gap:
            continue

        a_box = frame_bboxes[a_id]["bbox"]
        b_box = frame_bboxes[b_id]["bbox"]

        result = _fill_one_gap(
            model, frame_dir, seg_frames,
            a_id, b_id, a_box, b_box,
            img_w, img_h, resize_h=resize_h,
            min_visible_score=min_visible_score,
        )

        if result is not None:
            for fid, bbox in result.items():
                if fid in frame_bboxes:
                    frame_bboxes[fid]["bbox"] = bbox
                    frame_bboxes[fid]["cotracker"] = True
            replaced += gap_len
        else:
            failed += 1

    if replaced > 0 or failed > 0:
        print(f"  CoTracker3: replaced {replaced} gap frames "
              f"({failed} gap(s) fell back to LK fill)")


def _fill_one_gap(
    model: "CoTrackerPredictor",
    frame_dir: Path,
    seg_frames: list[dict],
    a_id: int,
    b_id: int,
    a_box: list[int],
    b_box: list[int],
    img_w: int,
    img_h: int,
    *,
    resize_h: int = 320,
    min_visible: int = 2,
    min_visible_score: float = 0.5,
) -> dict[int, list[int]] | None:
    """Fill one internal gap [a_id+1 … b_id-1] using CoTracker3.

    Returns {fid: [x1, y1, x2, y2]} or None if the inference fails.
    """
    import torch

    gap_len = b_id - a_id - 1
    if gap_len <= 0:
        return {}

    # Load frames from a_id to b_id inclusive
    frame_ids = list(range(a_id, b_id + 1))
    n_t = len(frame_ids)

    resize_w = max(1, int(img_w * resize_h / img_h))
    x_scale = resize_w / img_w
    y_scale = resize_h / img_h

    frames_list = []
    for fid in frame_ids:
        img_path = frame_dir / seg_frames[fid]["frame_file"]
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames_list.append(torch.from_numpy(img_rgb).float() / 255.0)

    # [1, T, 3, H, W]
    video = torch.stack(frames_list).permute(0, 3, 1, 2).unsqueeze(0)

    # Build query points from both anchors:
    # 5 from a_box at t=0 (corners + center)
    # 5 from b_box at t=n_t-1 (corners + center)
    ax1, ay1, ax2, ay2 = a_box
    bx1, by1, bx2, by2 = b_box

    def _box_queries(t_idx: int, box: list[int]) -> list[list[float]]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return [
            [t_idx, x1 * x_scale, y1 * y_scale],
            [t_idx, x2 * x_scale, y1 * y_scale],
            [t_idx, x1 * x_scale, y2 * y_scale],
            [t_idx, x2 * x_scale, y2 * y_scale],
            [t_idx, cx * x_scale, cy * y_scale],
        ]

    queries_raw = _box_queries(0, a_box) + _box_queries(n_t - 1, b_box)
    queries = torch.tensor([queries_raw], dtype=torch.float32)

    try:
        with torch.no_grad():
            pred_tracks, pred_vis = model(video, queries=queries)
        # pred_tracks: [1, T, N, 2]  (x, y in resize_h/resize_w space)
        # pred_vis:    [1, T, N]
    except Exception:
        return None

    result: dict[int, list[int]] = {}

    for t_idx, fid in enumerate(frame_ids):
        if fid == a_id or fid == b_id:
            continue  # keep anchor detections unchanged

        tracks_t = pred_tracks[0, t_idx]  # [N, 2]
        vis_t = pred_vis[0, t_idx]         # [N]

        vis_mask = vis_t > min_visible_score
        if int(vis_mask.sum()) < min_visible:
            return None  # too many points lost — trust LK fill more

        # Mean of visible points → center in resized space → unscale
        cx = float(tracks_t[vis_mask, 0].mean()) / x_scale
        cy = float(tracks_t[vis_mask, 1].mean()) / y_scale

        # Linearly interpolate bbox size between anchors
        t = (fid - a_id) / (b_id - a_id)
        w_a = ax2 - ax1
        h_a = ay2 - ay1
        w_b = bx2 - bx1
        h_b = by2 - by1
        w = int(round((1.0 - t) * w_a + t * w_b))
        h = int(round((1.0 - t) * h_a + t * h_b))

        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = x1 + max(1, w)
        y2 = y1 + max(1, h)

        # Clamp to image bounds
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))

        result[fid] = [x1, y1, x2, y2]

    return result
