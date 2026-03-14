"""Venue mapping via structure-masked feature matching + OF interpolation.

**Concept (user insight 2026-03-14)**
-------------------------------------
Throughout the run, at certain frames the athlete passes near distinctive
dark features (rocks, tree clusters) that appear in BOTH the video frame AND
the venue image.  At those frames, feature matching gives an **absolute** venue
position.  Between matched keyframes, background OF gives smooth **relative**
interpolation (like Phase B gap-fill in tracking).

**Algorithm**
--------------
1. For every keyframe (every ``keyframe_interval`` frames):
   a. Apply structure mask: keep only dark, textured pixels (rocks/trees).
   b. Run LoFTR feature matching between masked video frame and venue image.
   c. Validate homography geometry (convex quad, reasonable area ratio).
   d. If accepted: record homography H_t + projected athlete venue position.

2. Between accepted keyframes K_a and K_b:
   a. Compute cumulative background OF from K_a to each intermediate frame.
   b. Interpolate: v(t) = v(K_a) - OF_scale * cum_bg_flow(K_a → t)
      (OF_scale estimated from K_a and K_b positions + their OF displacements)
   c. Fallback: temporal linear interpolation between K_a and K_b.

3. Outside matched keyframe range: linear extrapolation from nearest anchor.

**Why this is valid (no per-run GT at inference)**
---------------------------------------------------
- Feature matching uses background texture, not athlete position.
- Background OF uses non-athlete pixels only.
- GT is used ONLY for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_tracking(manifest_path: Path) -> tuple[list[int], list[str], list[list[float]]]:
    """Return (frame_ids, frame_files, padded_bboxes)."""
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frame_ids, files, bboxes = [], [], []
    centers = {}
    for fr in data.get("frames", []):
        fname = fr.get("frame_file", "")
        try:
            fid = int(fname.replace("frame_", "").replace(".jpg", "").replace(".png", ""))
        except ValueError:
            fid = fr.get("frame_id", 0)
        frame_ids.append(fid)
        files.append(fname)
        bb = fr.get("bbox_padded") or fr.get("bbox") or [0, 0, 64, 64]
        bboxes.append(bb)
        bbox = fr.get("bbox")
        if bbox and len(bbox) == 4:
            centers[fid] = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
    return frame_ids, files, bboxes, centers


def _load_gt(gt_path: Path) -> dict[int, tuple[float, float]]:
    gt: dict[int, tuple[float, float]] = {}
    with gt_path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                gt[int(row[0])] = (float(row[1]), float(row[2]))
            except (ValueError, IndexError):
                continue
    return gt


# ---------------------------------------------------------------------------
# Structure masking
# ---------------------------------------------------------------------------

def make_structure_mask(
    img_bgr: np.ndarray,
    *,
    dark_threshold: int = 90,
    min_gradient: float = 8.0,
    min_area: int = 200,
) -> np.ndarray:
    """Return a binary mask of dark, textured pixels (rocks/trees).

    Excludes bright snow (V > threshold), textureless regions, and tiny blobs.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Snow / sky: bright OR low-saturation bright
    snow_sky = (hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 50)
    # Very bright anything
    bright = gray > dark_threshold

    non_feature = snow_sky | bright
    # Dilate to avoid edge artifacts
    kernel = np.ones((11, 11), np.uint8)
    non_feature = cv2.dilate(non_feature.astype(np.uint8), kernel).astype(bool)

    # Texture: gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    textured = grad > min_gradient

    raw_mask = (~non_feature) & textured

    # Remove small blobs
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw_mask.astype(np.uint8), connectivity=8
    )
    clean_mask = np.zeros_like(raw_mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == lbl] = True

    return clean_mask


# ---------------------------------------------------------------------------
# Feature matching (LoFTR with structure mask)
# ---------------------------------------------------------------------------

def _try_loftr_homography(
    matcher,
    frame_bgr: np.ndarray,
    venue_bgr: np.ndarray,
    *,
    device: str,
    max_side: int,
    min_inliers: int,
    ransac_threshold: float,
    dark_threshold: int = 90,
    min_gradient: float = 8.0,
    min_area: int = 200,
) -> tuple[np.ndarray | None, int, int, str]:
    """Run structure-masked LoFTR and return (H, n_matches, n_inliers, reason)."""
    import torch

    struct_mask = make_structure_mask(
        frame_bgr,
        dark_threshold=dark_threshold,
        min_gradient=min_gradient,
        min_area=min_area,
    )
    coverage = float(struct_mask.mean())

    if coverage < 0.01:
        return None, 0, 0, f"structure coverage too low ({coverage:.3f})"

    # Apply mask to frame (zero out non-structure pixels → LoFTR won't find keypoints there)
    masked_frame = frame_bgr.copy()
    masked_frame[~struct_mask] = 0

    def _to_tensor(img):
        h, w = img.shape[:2]
        scale = float(max_side) / float(max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        return t.to(device), scale if scale < 1.0 else 1.0

    frame_t, frame_sc = _to_tensor(masked_frame)
    venue_t, venue_sc = _to_tensor(venue_bgr)

    with torch.inference_mode():
        pred = matcher({"image0": frame_t, "image1": venue_t})

    mkpts0 = pred["keypoints0"].cpu().numpy()
    mkpts1 = pred["keypoints1"].cpu().numpy()
    conf = pred["confidence"].cpu().numpy()

    # Filter low-confidence matches (snow noise)
    ok = conf >= 0.5
    mkpts0, mkpts1 = mkpts0[ok], mkpts1[ok]

    n_matches = int(mkpts0.shape[0])
    if n_matches < 4:
        return None, n_matches, 0, f"too few matches ({n_matches})"

    # Scale back to original coords
    mkpts0 = mkpts0 / frame_sc
    mkpts1 = mkpts1 / venue_sc

    H, inlier_mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransac_threshold)
    if H is None or inlier_mask is None:
        return None, n_matches, 0, "RANSAC failed"

    n_inliers = int(inlier_mask.ravel().sum())
    if n_inliers < min_inliers:
        return None, n_matches, n_inliers, f"too few inliers ({n_inliers} < {min_inliers})"

    # Geometry validation: projected frame corners should be a reasonable quad
    fh, fw = frame_bgr.shape[:2]
    vh, vw = venue_bgr.shape[:2]
    corners = np.float32([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(corners, H).reshape(4, 2)

    # Check convexity
    hull = cv2.convexHull(dst.astype(np.float32))
    if len(hull) < 4:
        return None, n_matches, n_inliers, "non-convex projected quad"

    # Check area ratio (frame should be a small portion of venue)
    def _shoelace(pts):
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 0]
        return abs(area) * 0.5

    area_ratio = _shoelace(dst) / float(vw * vh)
    if area_ratio > 0.4 or area_ratio < 0.0005:
        return None, n_matches, n_inliers, f"area ratio out of range ({area_ratio:.4f})"

    return H.astype(np.float32), n_matches, n_inliers, "ok"


# ---------------------------------------------------------------------------
# Background OF for interpolation
# ---------------------------------------------------------------------------

def _compute_frame_bg_flow(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    athlete_bbox: list[float],
) -> tuple[float, float]:
    """One-step background flow (dark-feature weighted)."""
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr, None,
        0.5, 3, 15, 3, 5, 1.2, 0,
    )
    h, w = gray_prev.shape[:2]
    # Dark-feature mask
    dark = gray_prev < 90
    gx = cv2.Sobel(gray_prev, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray_prev, cv2.CV_32F, 0, 1)
    tex = np.sqrt(gx ** 2 + gy ** 2) > 8.0
    x1, y1, x2, y2 = [max(0, int(v)) for v in athlete_bbox[:4]]
    x2, y2 = min(w, x2), min(h, y2)
    ath = np.zeros((h, w), bool)
    if x2 > x1 and y2 > y1:
        ath[y1:y2, x1:x2] = True
    mask = dark & tex & ~ath
    if mask.sum() < 20:
        # Fallback: all non-athlete
        mask = ~ath
    if mask.sum() == 0:
        return 0.0, 0.0
    return float(np.median(flow[mask, 0])), float(np.median(flow[mask, 1]))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_venue_hybrid(
    video_path: str | Path,
    *,
    venue_image_path: str | Path = "venue_image.png",
    tracking_root: str | Path = "output/tracking",
    frame_root: str | Path = "output/frames",
    gt_root: str | Path = "annotations/venue",
    output_root: str | Path = "output/venue_mapping",
    keyframe_interval: int = 50,
    loftr_max_side: int = 840,
    min_inliers: int = 8,
    ransac_threshold: float = 8.0,
    dark_threshold: int = 90,
    min_gradient: float = 8.0,
    device: str = "mps",
    verbose: bool = True,
) -> dict:
    """Run hybrid venue mapping: LoFTR keyframes + OF interpolation."""
    video_path = Path(video_path)
    video_stem = video_path.stem

    frame_dir = Path(frame_root) / video_stem
    tracking_manifest = Path(tracking_root) / video_stem / "tracking.json"
    venue_image_path = Path(venue_image_path)
    gt_path = Path(gt_root) / video_stem / "gt_venue.csv"
    output_dir = Path(output_root) / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for p, label in [
        (frame_dir, "Frame directory"),
        (tracking_manifest, "Tracking manifest"),
        (venue_image_path, "Venue image"),
        (gt_path, "GT annotations"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    # Load LoFTR
    try:
        import torch
        from kornia.feature import LoFTR
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        matcher = LoFTR(pretrained="outdoor").to(device).eval()
        if verbose:
            print(f"[hybrid] LoFTR loaded on {device}")
    except ImportError:
        raise ImportError("kornia required. Install with: uv add kornia")

    venue_bgr = cv2.imread(str(venue_image_path), cv2.IMREAD_COLOR)
    if venue_bgr is None:
        raise RuntimeError(f"Cannot read venue image: {venue_image_path}")
    venue_h, venue_w = venue_bgr.shape[:2]

    frame_ids, frame_files, bboxes, centers = _load_tracking(tracking_manifest)
    gt = _load_gt(gt_path)
    n = len(frame_ids)

    if verbose:
        print(f"[hybrid] {video_stem}: {n} frames, {len(gt)} GT anchors")
        print(f"[hybrid] Trying LoFTR on keyframes (every {keyframe_interval} frames)...")

    # --- Phase 1: LoFTR on keyframes ---
    keyframe_positions: dict[int, tuple[float, float]] = {}  # frame_idx → venue pos

    keyframe_indices = list(range(0, n, keyframe_interval))
    if n - 1 not in keyframe_indices:
        keyframe_indices.append(n - 1)

    for ki in keyframe_indices:
        fname = frame_files[ki]
        frame_bgr = cv2.imread(str(frame_dir / fname))
        if frame_bgr is None:
            continue

        fh, fw = frame_bgr.shape[:2]
        H, n_matches, n_inliers, reason = _try_loftr_homography(
            matcher, frame_bgr, venue_bgr,
            device=device,
            max_side=loftr_max_side,
            min_inliers=min_inliers,
            ransac_threshold=ransac_threshold,
            dark_threshold=dark_threshold,
            min_gradient=min_gradient,
        )

        if H is not None:
            # Project athlete center through homography
            fid = frame_ids[ki]
            if fid in centers:
                cx, cy = centers[fid]
            else:
                cx, cy = fw / 2, fh / 2
            pt = np.float32([[[cx, cy]]])
            proj = cv2.perspectiveTransform(pt, H)[0, 0]
            vx = float(np.clip(proj[0], 0, venue_w - 1))
            vy = float(np.clip(proj[1], 0, venue_h - 1))
            keyframe_positions[ki] = (vx, vy)
            if verbose:
                print(f"  [hybrid] Keyframe {ki} (fid={fid}): ACCEPTED "
                      f"matches={n_matches} inliers={n_inliers} "
                      f"→ venue ({vx:.0f},{vy:.0f})")
        else:
            if verbose:
                print(f"  [hybrid] Keyframe {ki} (fid={frame_ids[ki]}): rejected ({reason})")

    n_accepted = len(keyframe_positions)
    if verbose:
        print(f"[hybrid] Accepted {n_accepted}/{len(keyframe_indices)} keyframes")

    if n_accepted == 0:
        if verbose:
            print("[hybrid] No keyframes accepted — LoFTR failed on all keyframes.")
        return {
            "video_stem": video_stem,
            "n_accepted_keyframes": 0,
            "error": "No LoFTR keyframes accepted",
        }

    # --- Phase 2: Background OF accumulation ---
    if verbose:
        print("[hybrid] Computing background OF for interpolation...")

    per_frame_bg = np.zeros((n, 2), dtype=float)
    prev_gray = None
    for i, fname in enumerate(frame_files):
        curr_gray = cv2.imread(str(frame_dir / fname), cv2.IMREAD_GRAYSCALE)
        if curr_gray is None:
            prev_gray = None
            continue
        if prev_gray is not None:
            dx, dy = _compute_frame_bg_flow(prev_gray, curr_gray, bboxes[i])
            per_frame_bg[i] = [dx, dy]
        prev_gray = curr_gray
        if verbose and i % 200 == 0:
            print(f"  [OF] {i}/{n}...", end="\r")

    if verbose:
        print(f"  [OF] {n}/{n} done.  ")

    # Cumulative OF from each point
    cum_flow = np.cumsum(per_frame_bg, axis=0)

    # --- Phase 3: Dense position estimation ---
    # For each frame, use nearest anchor keyframes + OF to estimate venue position
    positions: list[tuple[float, float]] = [(0.0, 0.0)] * n
    sorted_anchors = sorted(keyframe_positions.keys())

    for i in range(n):
        # Find bracketing anchors
        before = [k for k in sorted_anchors if k <= i]
        after = [k for k in sorted_anchors if k >= i]

        if not before and not after:
            positions[i] = (venue_w / 2, venue_h / 2)
            continue

        if i in keyframe_positions:
            positions[i] = keyframe_positions[i]
            continue

        if before and after:
            k_lo = before[-1]
            k_hi = after[0]
            # Interpolate using OF relative displacement
            # venue_pos(i) ≈ venue_pos(k_lo) - scale × (cum_flow[i] - cum_flow[k_lo])
            # Compute scale from both anchors:
            delta_venue = np.array(keyframe_positions[k_hi]) - np.array(keyframe_positions[k_lo])
            delta_flow = cum_flow[k_hi] - cum_flow[k_lo]
            # Avoid division by zero
            scale_x = delta_venue[0] / delta_flow[0] if abs(delta_flow[0]) > 0.1 else 0.0
            scale_y = delta_venue[1] / delta_flow[1] if abs(delta_flow[1]) > 0.1 else 0.0
            df_i = cum_flow[i] - cum_flow[k_lo]
            vx = keyframe_positions[k_lo][0] + scale_x * df_i[0]
            vy = keyframe_positions[k_lo][1] + scale_y * df_i[1]
            vx = float(np.clip(vx, 0, venue_w - 1))
            vy = float(np.clip(vy, 0, venue_h - 1))
            positions[i] = (vx, vy)

        elif before:
            # Extrapolate forward from last anchor
            k = before[-1]
            if len(before) >= 2:
                k_prev = before[-2]
                delta_venue = np.array(keyframe_positions[k]) - np.array(keyframe_positions[k_prev])
                delta_flow = cum_flow[k] - cum_flow[k_prev]
                scale_x = delta_venue[0] / delta_flow[0] if abs(delta_flow[0]) > 0.1 else 0.0
                scale_y = delta_venue[1] / delta_flow[1] if abs(delta_flow[1]) > 0.1 else 0.0
            else:
                scale_x, scale_y = -1.0, -1.0  # rough guess
            df_i = cum_flow[i] - cum_flow[k]
            vx = float(np.clip(keyframe_positions[k][0] + scale_x * df_i[0], 0, venue_w - 1))
            vy = float(np.clip(keyframe_positions[k][1] + scale_y * df_i[1], 0, venue_h - 1))
            positions[i] = (vx, vy)

        else:
            # Extrapolate backward from first anchor
            k = after[0]
            if len(after) >= 2:
                k_next = after[1]
                delta_venue = np.array(keyframe_positions[k_next]) - np.array(keyframe_positions[k])
                delta_flow = cum_flow[k_next] - cum_flow[k]
                scale_x = delta_venue[0] / delta_flow[0] if abs(delta_flow[0]) > 0.1 else 0.0
                scale_y = delta_venue[1] / delta_flow[1] if abs(delta_flow[1]) > 0.1 else 0.0
            else:
                scale_x, scale_y = -1.0, -1.0
            df_i = cum_flow[i] - cum_flow[k]
            vx = float(np.clip(keyframe_positions[k][0] + scale_x * df_i[0], 0, venue_w - 1))
            vy = float(np.clip(keyframe_positions[k][1] + scale_y * df_i[1], 0, venue_h - 1))
            positions[i] = (vx, vy)

    # --- Evaluate at GT frames ---
    id_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
    errors = []
    if verbose:
        print("\n[hybrid] Evaluation at GT frames (within tracking range):")
    for fid, (true_vx, true_vy) in sorted(gt.items()):
        if fid not in id_to_idx:
            continue
        idx = id_to_idx[fid]
        pred_vx, pred_vy = positions[idx]
        err = float(np.sqrt((pred_vx - true_vx) ** 2 + (pred_vy - true_vy) ** 2))
        errors.append(err)
        if verbose:
            print(f"  frame {fid:5d}: true=({true_vx:.0f},{true_vy:.0f})  "
                  f"pred=({pred_vx:.0f},{pred_vy:.0f})  err={err:.1f}px")

    results = {}
    if errors:
        errors_arr = np.array(errors)
        results = {
            "n_accepted_keyframes": n_accepted,
            "keyframe_indices": sorted_anchors,
            "mean_err_px": float(np.mean(errors_arr)),
            "median_err_px": float(np.median(errors_arr)),
            "p90_err_px": float(np.percentile(errors_arr, 90)),
            "within_50px": float(np.mean(errors_arr <= 50.0)),
        }
        if verbose:
            print(f"\n[hybrid] Summary: "
                  f"mean={results['mean_err_px']:.1f}px  "
                  f"median={results['median_err_px']:.1f}px  "
                  f"P90={results['p90_err_px']:.1f}px  "
                  f"within50={results['within_50px']:.0%}")

    # Save
    out = {
        "video_stem": video_stem,
        "n_tracking_frames": n,
        "n_gt_anchors": len(gt),
        "accepted_keyframes": {
            str(frame_ids[k]): list(v)
            for k, v in keyframe_positions.items()
        },
        **results,
    }
    out_path = output_dir / "hybrid_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if verbose:
        print(f"[hybrid] Results saved: {out_path}")
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid venue mapping: LoFTR keyframes + OF interpolation.")
    p.add_argument("video", type=Path)
    p.add_argument("--venue-image", type=Path, default=Path("venue_image.png"))
    p.add_argument("--tracking-root", type=Path, default=Path("output/tracking"))
    p.add_argument("--frame-root", type=Path, default=Path("output/frames"))
    p.add_argument("--gt-root", type=Path, default=Path("annotations/venue"))
    p.add_argument("--output-root", type=Path, default=Path("output/venue_mapping"))
    p.add_argument("--keyframe-interval", type=int, default=50)
    p.add_argument("--loftr-max-side", type=int, default=840)
    p.add_argument("--min-inliers", type=int, default=8)
    p.add_argument("--ransac-threshold", type=float, default=8.0)
    p.add_argument("--dark-threshold", type=int, default=90)
    p.add_argument("--min-gradient", type=float, default=8.0)
    p.add_argument("--device", type=str, default="mps")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_venue_hybrid(
        args.video,
        venue_image_path=args.venue_image,
        tracking_root=args.tracking_root,
        frame_root=args.frame_root,
        gt_root=args.gt_root,
        output_root=args.output_root,
        keyframe_interval=args.keyframe_interval,
        loftr_max_side=args.loftr_max_side,
        min_inliers=args.min_inliers,
        ransac_threshold=args.ransac_threshold,
        dark_threshold=args.dark_threshold,
        min_gradient=args.min_gradient,
        device=args.device,
    )


if __name__ == "__main__":
    main()
