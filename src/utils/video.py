"""Frame extraction from raw video files."""

from pathlib import Path

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

    Returns
    -------
    List of paths to the extracted frame images.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
