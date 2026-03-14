"""Venue mapping via multi-scale gradient template matching.

For each candidate keyframe, extract the background gradient image (excluding
the athlete bbox), then search exhaustively for that patch in the venue image
at multiple scales.  The intuition:

- Gradient magnitude is near-zero on uniform snow/sky and high on rocks/trees.
- The same distinctive structures (rocks, tree lines, ridge shapes) appear in
  both the telephoto video frame and the wide-angle venue image.
- By trying a range of scale factors we cover the full zoom range of the broadcast
  camera without needing to know zoom a priori.

Usage
-----
::
    uv run python scripts/venue_template.py \\
        "raw_videos/Snowboard Men_1_80_Fabian Knözinger.mp4" --no-video
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers  (shared with venue_camflow.py)
# ---------------------------------------------------------------------------

def _load_tracking(manifest_path: Path) -> tuple[list[int], list[str], list[list[float]]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frame_ids, files, bboxes, centers = [], [], [], []
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
        x1, y1, x2, y2 = bb
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
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
# Template matching
# ---------------------------------------------------------------------------

def _gradient_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def _mask_athlete(img: np.ndarray, bbox: list[float]) -> np.ndarray:
    """Zero out athlete bbox in-place on a copy."""
    out = img.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = 0.0
    return out


def find_frame_in_venue(
    frame_gray: np.ndarray,
    venue_grad: np.ndarray,
    bbox: list[float],
    *,
    scale_min: float = 0.03,
    scale_max: float = 0.20,
    n_scales: int = 30,
    ncc_threshold: float = 0.25,
    athlete_center: tuple[float, float] | None = None,
) -> dict | None:
    """Search for the video frame in the venue image at multiple scales.

    The gradient of the background (athlete masked out) is used as the template.
    NCC between the resized template and the venue gradient finds the best match.

    Parameters
    ----------
    frame_gray : np.ndarray
        Grayscale video frame.
    venue_grad : np.ndarray
        Pre-computed gradient magnitude of the venue image (float32).
    bbox : list[float]
        Athlete padded bbox [x1,y1,x2,y2] in frame pixels.
    scale_min / scale_max : float
        Range of (venue_pixels / frame_pixels) scale factors to search.
    n_scales : int
        Number of scale steps (log-spaced).
    ncc_threshold : float
        Minimum NCC score to accept a match.
    athlete_center : tuple | None
        Athlete center in frame pixels (cx, cy).  Used to compute venue
        position of the athlete from the matched frame region.

    Returns
    -------
    dict or None
        Keys: venue_cx, venue_cy (athlete venue position),
              frame_cx_in_venue, frame_cy_in_venue (frame center venue pos),
              scale, ncc, template_w, template_h
    """
    fh, fw = frame_gray.shape[:2]
    vh, vw = venue_grad.shape[:2]

    # Build background gradient template (athlete masked)
    frame_grad = _gradient_mag(frame_gray)
    frame_grad = _mask_athlete(frame_grad, bbox)

    if frame_grad.max() < 1e-6:
        return None

    best: dict | None = None
    best_ncc = -1.0

    for s in np.geomspace(scale_min, scale_max, n_scales):
        tw = int(round(fw * s))
        th = int(round(fh * s))
        if tw < 16 or th < 16:
            continue
        if tw >= vw or th >= vh:
            continue

        # Resize background gradient to template size
        tmpl = cv2.resize(frame_grad, (tw, th), interpolation=cv2.INTER_AREA)
        if tmpl.max() < 1e-6:
            continue

        # Normalize template to avoid brightness bias
        tmpl = tmpl.astype(np.float32)

        # Crop venue_grad to valid search region
        res = cv2.matchTemplate(venue_grad, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_ncc:
            best_ncc = max_val
            # top-left of matched template in venue → frame center in venue
            frame_cx_v = max_loc[0] + tw / 2.0
            frame_cy_v = max_loc[1] + th / 2.0

            # Athlete offset from frame center → athlete position in venue
            if athlete_center is not None:
                acx, acy = athlete_center
                offset_x = (acx - fw / 2.0) * s
                offset_y = (acy - fh / 2.0) * s
                venue_cx = frame_cx_v + offset_x
                venue_cy = frame_cy_v + offset_y
            else:
                venue_cx = frame_cx_v
                venue_cy = frame_cy_v

            best = {
                "venue_cx": venue_cx,
                "venue_cy": venue_cy,
                "frame_cx_in_venue": frame_cx_v,
                "frame_cy_in_venue": frame_cy_v,
                "scale": s,
                "ncc": max_val,
                "template_w": tw,
                "template_h": th,
            }

    if best is None or best["ncc"] < ncc_threshold:
        return None
    return best


def run_template_search(
    frame_dir: Path,
    frame_files: list[str],
    frame_ids: list[int],
    bboxes: list[list[float]],
    centers: list[tuple[float, float]],
    venue_grad: np.ndarray,
    *,
    keyframe_step: int = 50,
    scale_min: float = 0.03,
    scale_max: float = 0.20,
    n_scales: int = 30,
    ncc_threshold: float = 0.25,
    verbose: bool = True,
) -> list[dict]:
    """Run template matching on every `keyframe_step`-th frame.

    Returns a list of accepted matches sorted by frame_id.
    """
    accepted = []
    n = len(frame_files)
    candidate_indices = list(range(0, n, keyframe_step))

    for i, idx in enumerate(candidate_indices):
        fname = frame_files[idx]
        fid = frame_ids[idx]
        frame_bgr = cv2.imread(str(frame_dir / fname), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        result = find_frame_in_venue(
            gray, venue_grad, bboxes[idx],
            scale_min=scale_min, scale_max=scale_max, n_scales=n_scales,
            ncc_threshold=ncc_threshold,
            athlete_center=centers[idx],
        )

        if result is not None:
            result["frame_id"] = fid
            result["frame_idx"] = idx
            accepted.append(result)
            if verbose:
                print(f"  [tmpl] frame {fid:5d}: ncc={result['ncc']:.3f}  "
                      f"scale={result['scale']:.3f}  "
                      f"venue=({result['venue_cx']:.0f},{result['venue_cy']:.0f})")
        else:
            if verbose:
                print(f"  [tmpl] frame {fid:5d}: no match (best ncc below threshold)")

        if verbose and i % 10 == 0:
            print(f"  [tmpl] {i+1}/{len(candidate_indices)} candidates processed", end="\r")

    if verbose:
        print(f"\n[tmpl] {len(accepted)} / {len(candidate_indices)} keyframes matched")
    return accepted


# ---------------------------------------------------------------------------
# Evaluation against GT
# ---------------------------------------------------------------------------

def evaluate_matches(
    matches: list[dict],
    gt: dict[int, tuple[float, float]],
    frame_ids: list[int],
    *,
    verbose: bool = True,
) -> dict:
    """Compare matched venue positions to GT for frames that have GT."""
    id_to_match = {m["frame_id"]: m for m in matches}
    errors = []
    details = []

    for fid, (gt_vx, gt_vy) in sorted(gt.items()):
        # Find nearest matched frame within ±25 frames
        nearby = [(abs(fid - m["frame_id"]), m) for m in matches
                  if abs(fid - m["frame_id"]) <= 25]
        if not nearby:
            continue
        nearest_match = min(nearby, key=lambda x: x[0])[1]
        pred_vx = nearest_match["venue_cx"]
        pred_vy = nearest_match["venue_cy"]
        err = float(np.sqrt((pred_vx - gt_vx) ** 2 + (pred_vy - gt_vy) ** 2))
        errors.append(err)
        details.append({
            "gt_frame": fid,
            "match_frame": nearest_match["frame_id"],
            "true": (gt_vx, gt_vy),
            "pred": (pred_vx, pred_vy),
            "err_px": err,
            "ncc": nearest_match["ncc"],
        })
        if verbose:
            print(f"  GT frame {fid:5d} → match frame {nearest_match['frame_id']:5d}: "
                  f"true=({gt_vx:.0f},{gt_vy:.0f})  pred=({pred_vx:.0f},{pred_vy:.0f})  "
                  f"err={err:.1f}px  ncc={nearest_match['ncc']:.3f}")

    if not errors:
        return {"n_evaluated": 0, "mean_err_px": None}

    errors_arr = np.array(errors)
    return {
        "n_evaluated": len(errors),
        "mean_err_px": float(np.mean(errors_arr)),
        "median_err_px": float(np.median(errors_arr)),
        "p90_err_px": float(np.percentile(errors_arr, 90)),
        "within_50px": float(np.mean(errors_arr <= 50.0)),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venue_template(
    video_path: str | Path,
    *,
    venue_image_path: str | Path = "venue_image.png",
    tracking_root: str | Path = "output/tracking",
    frame_root: str | Path = "output/frames",
    gt_root: str | Path = "annotations/venue",
    output_root: str | Path = "output/venue_mapping",
    keyframe_step: int = 50,
    scale_min: float = 0.03,
    scale_max: float = 0.20,
    n_scales: int = 30,
    ncc_threshold: float = 0.25,
    no_video: bool = False,
    verbose: bool = True,
) -> dict:
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

    venue_bgr = cv2.imread(str(venue_image_path), cv2.IMREAD_COLOR)
    if venue_bgr is None:
        raise RuntimeError(f"Cannot read venue image: {venue_image_path}")
    venue_h, venue_w = venue_bgr.shape[:2]

    frame_ids, frame_files, bboxes, centers = _load_tracking(tracking_manifest)
    gt = _load_gt(gt_path)

    if verbose:
        print(f"[tmpl] {video_stem}: {len(frame_ids)} frames, {len(gt)} GT anchors")
        print(f"[tmpl] Pre-computing venue gradient...")

    venue_gray = cv2.cvtColor(venue_bgr, cv2.COLOR_BGR2GRAY)
    venue_grad = _gradient_mag(venue_gray).astype(np.float32)

    if verbose:
        print(f"[tmpl] Searching {len(range(0, len(frame_files), keyframe_step))} "
              f"candidate frames (every {keyframe_step} frames), "
              f"scales=[{scale_min:.2f}, {scale_max:.2f}] n={n_scales}...")

    matches = run_template_search(
        frame_dir, frame_files, frame_ids, bboxes, centers,
        venue_grad,
        keyframe_step=keyframe_step,
        scale_min=scale_min,
        scale_max=scale_max,
        n_scales=n_scales,
        ncc_threshold=ncc_threshold,
        verbose=verbose,
    )

    if verbose:
        print(f"\n[tmpl] Evaluating against GT...")
    eval_res = evaluate_matches(matches, gt, frame_ids, verbose=verbose)

    if verbose and eval_res["n_evaluated"] > 0:
        print(f"\n[tmpl] vs GT: mean={eval_res['mean_err_px']:.1f}px  "
              f"median={eval_res['median_err_px']:.1f}px  "
              f"P90={eval_res['p90_err_px']:.1f}px  "
              f"within50={eval_res['within_50px']:.0%}  "
              f"n={eval_res['n_evaluated']}")

    # Save matched positions
    results = {
        "video_stem": video_stem,
        "n_tracking_frames": len(frame_ids),
        "n_gt_anchors": len(gt),
        "keyframe_step": keyframe_step,
        "scale_range": [scale_min, scale_max],
        "n_scales": n_scales,
        "ncc_threshold": ncc_threshold,
        "n_matches": len(matches),
        "matches": matches,
        "eval_vs_gt": eval_res,
    }
    out_path = output_dir / "template_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if verbose:
        print(f"[tmpl] Results saved: {out_path}")

    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Venue mapping via multi-scale template matching.")
    p.add_argument("video", type=Path)
    p.add_argument("--venue-image", type=Path, default=Path("venue_image.png"))
    p.add_argument("--tracking-root", type=Path, default=Path("output/tracking"))
    p.add_argument("--frame-root", type=Path, default=Path("output/frames"))
    p.add_argument("--gt-root", type=Path, default=Path("annotations/venue"))
    p.add_argument("--output-root", type=Path, default=Path("output/venue_mapping"))
    p.add_argument("--keyframe-step", type=int, default=50)
    p.add_argument("--scale-min", type=float, default=0.03)
    p.add_argument("--scale-max", type=float, default=0.20)
    p.add_argument("--n-scales", type=int, default=30)
    p.add_argument("--ncc-threshold", type=float, default=0.25)
    p.add_argument("--no-video", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_venue_template(
        args.video,
        venue_image_path=args.venue_image,
        tracking_root=args.tracking_root,
        frame_root=args.frame_root,
        gt_root=args.gt_root,
        output_root=args.output_root,
        keyframe_step=args.keyframe_step,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        n_scales=args.n_scales,
        ncc_threshold=args.ncc_threshold,
        no_video=args.no_video,
    )


if __name__ == "__main__":
    main()
