"""Ski detection using segmentation models.

Two strategies:
1. Grounding DINO + SAM2 — prompt-based zero-shot detection ("ski").
2. Color segmentation fallback — threshold on ski colour in HSV space.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def detect_skis(
    frame_dir: str | Path,
    output_dir: str | Path,
    *,
    method: str = "color_segmentation",
    prompt: str = "ski",
    device: str = "mps",
    confidence: float = 0.35,
) -> Path:
    """Detect skis in each frame.

    Returns a JSON file with per-frame bounding boxes / masks for the skis.
    """
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(
        list(frame_dir.glob("frame_*.jpg"))
        + list(frame_dir.glob("frame_*.png"))
    )
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    print(f"Detecting skis in {len(frame_paths)} frames (method={method})")

    if method == "grounding_sam2":
        detections = _detect_grounding_sam2(
            frame_paths, output_dir, prompt=prompt, device=device, confidence=confidence
        )
    else:
        detections = _detect_color_segmentation(frame_paths, output_dir)

    out_path = output_dir / "ski_detections.json"
    with open(out_path, "w") as f:
        json.dump({"method": method, "frames": detections}, f, indent=2)

    print(f"  → Ski detections saved to {out_path}")
    return out_path


def _detect_grounding_sam2(
    frame_paths: list[Path],
    output_dir: Path,
    *,
    prompt: str = "ski",
    device: str = "mps",
    confidence: float = 0.35,
) -> list[dict]:
    """Detect skis using GroundingDINO + SAM2."""
    try:
        # This requires groundingdino and sam2 packages
        from groundingdino.util.inference import load_model, predict
        # SAM2 for mask refinement
    except ImportError:
        print(
            "⚠  GroundingDINO/SAM2 not installed — falling back to colour segmentation.\n"
            "   Install with:  uv add groundingdino-py segment-anything-2"
        )
        return _detect_color_segmentation(frame_paths, output_dir)

    # TODO: Load GroundingDINO model and SAM2
    # model = load_model(config_path, checkpoint_path)
    # sam = SAM2ImagePredictor(...)

    detections: list[dict] = []
    for idx, fpath in enumerate(tqdm(frame_paths, desc="Ski detection (GDINO)")):
        # Placeholder structure
        detections.append(
            {
                "frame_id": idx,
                "frame_file": fpath.name,
                "ski_bboxes": [],
                "ski_mask_file": None,
            }
        )
    return detections


def _detect_color_segmentation(
    frame_paths: list[Path],
    output_dir: Path,
) -> list[dict]:
    """Simple colour-based ski segmentation (HSV thresholding).

    Works best when skis have distinctive colours (many competition skis do).
    This is a basic baseline — tune the HSV ranges for your specific footage.
    """
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    detections: list[dict] = []

    for idx, fpath in enumerate(tqdm(frame_paths, desc="Ski detection (colour)")):
        img = cv2.imread(str(fpath))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Look for elongated bright objects in the lower portion of the frame
        # These HSV ranges are a starting point — adjust per video
        # Range 1: bright whites/greys (many skis)
        lower1 = np.array([0, 0, 180])
        upper1 = np.array([180, 50, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)

        # Range 2: bright saturated colours (coloured skis)
        lower2 = np.array([0, 80, 150])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours and filter for elongated shapes (skis are long & thin)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ski_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if min(w, h) < 1:
                continue
            aspect_ratio = max(w, h) / min(w, h)
            # Skis have high aspect ratio (> 4:1)
            if aspect_ratio > 4.0:
                x, y, bw, bh = cv2.boundingRect(cnt)
                ski_bboxes.append([int(x), int(y), int(x + bw), int(y + bh)])

        # Save mask
        mask_path = mask_dir / f"ski_mask_{idx:06d}.png"
        cv2.imwrite(str(mask_path), mask)

        detections.append(
            {
                "frame_id": idx,
                "frame_file": fpath.name,
                "ski_bboxes": ski_bboxes,
                "ski_mask_file": str(mask_path.relative_to(output_dir)),
            }
        )

    return detections
