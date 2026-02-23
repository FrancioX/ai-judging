"""Frame extraction from raw video files."""

from pathlib import Path
import json
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
    fmt: str = "jpg",
    quality: int = 95,
    trim_start_seconds: float = 0,
    trim_end_seconds: float = 0,
) -> list[Path]:
    """Extract frames from a video file.

    Parameters
    ----------
    video_path : path to the input video.
    output_dir : directory where frames will be saved.
    fps : target FPS (None keeps the original frame rate).
    max_frames : cap on the number of extracted frames.
    fmt : image format (jpg, png).
    quality : JPEG quality (1-100).
    trim_start_seconds : number of seconds to trim from the start.
    trim_end_seconds : number of seconds to trim from the end.

    Returns
    -------
    List of paths to the extracted frame images.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Reuse existing frames if the directory already contains extracted images
    existing = sorted(output_dir.glob("frame_*.jpg")) + sorted(output_dir.glob("frame_*.png"))
    if existing:
        print(f"  → Reusing {len(existing)} existing frames in {output_dir}")
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / src_fps

    # Calculate frame indices for trimming
    start_frame = int(trim_start_seconds * src_fps)
    end_frame = max(start_frame + 1, int((total_duration - trim_end_seconds) * src_fps))
    end_frame = min(end_frame, total_frames)

    # Compute frame step for downsampling
    step = 1
    if fps is not None and fps < src_fps:
        step = max(1, round(src_fps / fps))

    saved: list[Path] = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=f"Extracting {video_path.stem}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames outside the trim range
        if frame_idx < start_frame or frame_idx >= end_frame:
            frame_idx += 1
            pbar.update(1)
            continue

        if frame_idx % step == 0:
            out_path = output_dir / f"frame_{frame_idx:06d}.{fmt}"
            params = (
                [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else []
            )
            cv2.imwrite(str(out_path), frame, params)
            saved.append(out_path)

            if max_frames and len(saved) >= max_frames:
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if trim_start_seconds > 0 or trim_end_seconds > 0:
        print(f"  → Trimmed {trim_start_seconds}s from start, {trim_end_seconds}s from end")
    print(f"  → Saved {len(saved)} frames to {output_dir}")
    return saved


def load_frames_as_array(
    frame_dir: str | Path,
    *,
    max_frames: int | None = None,
) -> np.ndarray:
    """Load extracted frames back into a (N, H, W, 3) numpy array (RGB)."""
    frame_dir = Path(frame_dir)
    paths = sorted(frame_dir.glob("frame_*.jpg")) + sorted(
        frame_dir.glob("frame_*.png")
    )
    if max_frames:
        paths = paths[:max_frames]
    frames = []
    for p in tqdm(paths, desc="Loading frames"):
        img = cv2.imread(str(p))
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def reconstruct_video_with_bboxes(
    frame_dir: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    fps: float = 30.0,
    bbox_color: tuple[int, int, int] = (0, 255, 0),
    bbox_thickness: int = 2,
) -> Path:
    """Reconstruct video from frames with bounding boxes overlaid.

    Parameters
    ----------
    frame_dir : directory containing extracted frames.
    manifest_path : path to segmentation.json with bbox info.
    output_path : output video file path.
    fps : frames per second for output video.
    bbox_color : BGR color tuple for the bounding box.
    bbox_thickness : thickness of the bounding box lines.

    Returns
    -------
    Path to the created video file.
    """
    frame_dir = Path(frame_dir)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the segmentation manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    frame_data = {fd["frame_id"]: fd for fd in manifest["frames"]}

    # Get frame paths in order
    frame_paths = sorted(
        list(frame_dir.glob("frame_*.jpg"))
        + list(frame_dir.glob("frame_*.png"))
    )

    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_paths[0]))
    h, w = first_frame.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"Reconstructing video with bounding boxes at {fps} fps...")

    for idx, fpath in enumerate(tqdm(frame_paths, desc="Rendering video")):
        img = cv2.imread(str(fpath))

        # Draw bounding box if detection exists
        if idx in frame_data and frame_data[idx].get("detected", False):
            bbox = frame_data[idx]["bbox_padded"]  # Use padded bbox
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, bbox_thickness)

            # Add confidence text
            conf = frame_data[idx].get("confidence", 0.0)
            label = f"Skier: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                img,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                bbox_color,
                -1,
            )

            # Draw text
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

        out.write(img)

    out.release()
    print(f"  → Video saved to {output_path}")
    return output_path
