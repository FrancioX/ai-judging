"""YOLO instance segmentation to isolate the skier from the background.

Uses Ultralytics YOLOv11-seg to detect and segment persons in each frame.
Runs with ``model.track()`` to assign persistent track IDs via ByteTrack.

Outputs per-frame:
  - Cropped frames (tight around the skier bounding box with padding)
  - Binary masks (pixel-level person segmentation)
  - A JSON manifest with bounding boxes, track IDs and all detected persons

This runs *before* the tracking stage and 2D pose estimation so
downstream stages can select the best track and smooth bounding boxes.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def segment_skier(
    frame_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "yolo11x-seg",
    device: str = "mps",
    confidence: float = 0.5,
    person_class_id: int = 0,
    padding_ratio: float = 0.15,
    select_strategy: str = "center",
) -> Path:
    """Run YOLO instance segmentation with tracking to isolate the skier.

    Uses ``model.track(persist=True)`` to get stable track IDs across
    frames via ByteTrack.  All detected persons and their track IDs are
    stored in the manifest so the downstream tracking stage can select
    the best track, smooth bounding boxes and fill detection gaps.

    Parameters
    ----------
    frame_dir : directory containing extracted frames (frame_*.jpg).
    output_dir : base output directory for segmentation results.
    model_name : YOLO model variant (yolo11{n,s,m,l,x}-seg).
    device : torch device ('mps', 'cuda:0', 'cpu').
    confidence : minimum detection confidence.
    person_class_id : COCO class ID for 'person' (always 0).
    padding_ratio : extra padding around the bbox as a fraction of size.
    select_strategy : how to pick the main skier when multiple persons
        are detected — 'largest' (biggest bbox area), 'center' (closest
        to frame center), or 'highest_conf'.

    Returns
    -------
    Path to the segmentation manifest JSON.
    """
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    crop_dir = output_dir / "crops"
    mask_dir = output_dir / "masks"
    crop_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(
        list(frame_dir.glob("frame_*.jpg"))
        + list(frame_dir.glob("frame_*.png"))
    )
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    print(f"Running YOLO segmentation + tracking on {len(frame_paths)} frames …")
    print(f"  Model: {model_name}  |  Device: {device}  |  Conf: {confidence}")
    print(f"  Strategy: {select_strategy}  |  Padding: {padding_ratio:.0%}")

    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
    except ImportError:
        print(
            "⚠  ultralytics not installed — writing stub segmentation.\n"
            "   Install with:  uv pip install ultralytics"
        )
        return _write_stub_segmentation(frame_paths, output_dir)

    manifest_frames: list[dict] = []

    for idx, fpath in enumerate(tqdm(frame_paths, desc="YOLO Seg+Track")):
        img = cv2.imread(str(fpath))
        h, w = img.shape[:2]

        # Run inference with tracking (persist=True keeps state across frames)
        results = model.track(
            str(fpath),
            conf=confidence,
            classes=[person_class_id],
            device=device,
            verbose=False,
            persist=True,
            tracker="bytetrack.yaml",
        )
        result = results[0]

        # Extract all person detections with track IDs
        persons = _extract_persons(result, h, w)

        if not persons:
            # No person detected — copy the original frame as-is
            crop_path = crop_dir / f"crop_{idx:06d}.jpg"
            mask_path = mask_dir / f"mask_{idx:06d}.png"
            cv2.imwrite(str(crop_path), img)
            cv2.imwrite(str(mask_path), np.zeros((h, w), dtype=np.uint8))
            manifest_frames.append({
                "frame_id": idx,
                "frame_file": fpath.name,
                "detected": False,
                "bbox": [0, 0, w, h],
                "bbox_padded": [0, 0, w, h],
                "confidence": 0.0,
                "crop_file": crop_path.name,
                "mask_file": mask_path.name,
                "persons": [],
            })
            continue

        # Select the main skier (used for backward-compatible crop output)
        person = _select_person(persons, h, w, strategy=select_strategy)

        # Compute padded bounding box
        x1, y1, x2, y2 = person["bbox"]
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)

        # Crop the image
        crop = img[py1:py2, px1:px2]
        crop_path = crop_dir / f"crop_{idx:06d}.jpg"
        cv2.imwrite(str(crop_path), crop)

        # Save the binary mask (full frame size)
        mask = person.get("mask", np.zeros((h, w), dtype=np.uint8))
        mask_path = mask_dir / f"mask_{idx:06d}.png"
        cv2.imwrite(str(mask_path), mask)

        # Build serialisable persons list (excludes numpy mask)
        persons_json = [
            {
                "bbox": p["bbox"],
                "confidence": float(p["confidence"]),
                "area": p["area"],
                "track_id": p["track_id"],
            }
            for p in persons
        ]

        manifest_frames.append({
            "frame_id": idx,
            "frame_file": fpath.name,
            "detected": True,
            "bbox": [x1, y1, x2, y2],
            "bbox_padded": [px1, py1, px2, py2],
            "confidence": float(person["confidence"]),
            "track_id": person["track_id"],
            "crop_file": crop_path.name,
            "mask_file": mask_path.name,
            "n_persons_detected": len(persons),
            "persons": persons_json,
        })

    # Write manifest
    manifest = {
        "model": model_name,
        "confidence_threshold": confidence,
        "select_strategy": select_strategy,
        "padding_ratio": padding_ratio,
        "tracking_enabled": True,
        "tracker": "bytetrack",
        "n_frames": len(manifest_frames),
        "frames": manifest_frames,
    }
    manifest_path = output_dir / "segmentation.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_detected = sum(1 for fr in manifest_frames if fr["detected"])
    print(f"  → Skier detected in {n_detected}/{len(manifest_frames)} frames")
    print(f"  → Crops saved to {crop_dir}")
    print(f"  → Manifest saved to {manifest_path}")
    return manifest_path


def _extract_persons(result, img_h: int, img_w: int) -> list[dict]:
    """Extract person detections (with track IDs) from a YOLO result object."""
    persons = []
    boxes = result.boxes
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        return persons

    # Track IDs assigned by ByteTrack (may be None on first appearance)
    track_ids = boxes.id
    has_ids = track_ids is not None

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if cls_id != 0:  # 0 = person in COCO
            continue

        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()

        tid: int | None = None
        if has_ids and i < len(track_ids):
            tid = int(track_ids[i].item())

        person = {
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "area": (x2 - x1) * (y2 - y1),
            "track_id": tid,
        }

        # Extract pixel mask if available
        if masks is not None and i < len(masks):
            mask_tensor = masks.data[i].cpu().numpy()
            # Resize mask to original image dimensions
            mask_resized = cv2.resize(
                mask_tensor.astype(np.float32),
                (img_w, img_h),
                interpolation=cv2.INTER_LINEAR,
            )
            person["mask"] = (mask_resized > 0.5).astype(np.uint8) * 255

        persons.append(person)

    return persons


def _select_person(
    persons: list[dict],
    img_h: int,
    img_w: int,
    *,
    strategy: str = "largest",
) -> dict:
    """Select the main skier from multiple person detections."""
    if len(persons) == 1:
        return persons[0]

    if strategy == "largest":
        return max(persons, key=lambda p: p["area"])

    elif strategy == "center":
        cx, cy = img_w / 2, img_h / 2
        def dist_to_center(p):
            bx = (p["bbox"][0] + p["bbox"][2]) / 2
            by = (p["bbox"][1] + p["bbox"][3]) / 2
            return (bx - cx) ** 2 + (by - cy) ** 2
        return min(persons, key=dist_to_center)

    elif strategy == "highest_conf":
        return max(persons, key=lambda p: p["confidence"])

    else:
        return max(persons, key=lambda p: p["area"])


def _write_stub_segmentation(
    frame_paths: list[Path], output_dir: Path
) -> Path:
    """Write a pass-through stub when ultralytics is not installed."""
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    manifest_frames = []
    for idx, fpath in enumerate(frame_paths):
        # Symlink or copy the original frame as the "crop"
        crop_path = crop_dir / f"crop_{idx:06d}.jpg"
        if not crop_path.exists():
            import shutil
            shutil.copy2(fpath, crop_path)

        manifest_frames.append({
            "frame_id": idx,
            "frame_file": fpath.name,
            "detected": False,
            "bbox": [0, 0, 0, 0],
            "bbox_padded": [0, 0, 0, 0],
            "confidence": 0.0,
            "crop_file": crop_path.name,
            "mask_file": None,
        })

    manifest = {
        "model": "stub",
        "n_frames": len(manifest_frames),
        "frames": manifest_frames,
    }
    manifest_path = output_dir / "segmentation.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path
