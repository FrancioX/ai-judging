"""3D pose lifting from 2D keypoint sequences.

Uses MotionBERT (Zhu et al., 2023) to lift 2D poses into 3D.
MotionBERT uses a temporal transformer that ingests a window of 2D poses
and outputs 3D coordinates in camera-relative space.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from urllib.request import urlretrieve

import numpy as np

from src.pose_3d.keypoint_converter import (
    H36M_KEYPOINTS_17,
    convert_keypoints_to_h36m_17,
    get_hip_indices,
)


def load_2d_poses(poses_2d_path: str | Path) -> tuple[np.ndarray, list[str], list[int]]:
    """Load 2D pose manifest into arrays and metadata.

    Returns
    -------
    tuple[np.ndarray, list[str], list[int]]
        (keypoints, keypoint_names, frame_ids) where keypoints has shape
        (N, 17, 3) in [x, y, confidence] format.
    """
    with open(poses_2d_path) as f:
        data = json.load(f)
    frames = data["frames"]
    keypoints = np.array([f["keypoints"] for f in frames], dtype=np.float32)
    keypoint_names = data.get("keypoint_names", [])
    frame_ids = [int(f.get("frame_id", i)) for i, f in enumerate(frames)]
    return keypoints, keypoint_names, frame_ids


def lift_to_3d(
    poses_2d_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "motionbert",
    checkpoint_path: str | Path | None = None,
    checkpoint_url: str | None = None,
    device: str = "mps",
    receptive_field: int = 243,
    model_kwargs: dict[str, Any] | None = None,
    batch_size: int = 16,
) -> Path:
    """Lift 2D poses to 3D using a temporal model.

    Parameters
    ----------
    poses_2d_path : path to the poses_2d.json produced by the 2D stage.
    output_dir : directory for output files.
    model_name : lifting model to use.
    checkpoint_path : path to model checkpoint.
    checkpoint_url : optional URL to download checkpoint when path is missing.
    device : torch device string.
    receptive_field : temporal window for MotionBERT.
    model_kwargs : optional kwargs used to instantiate DSTformer when loading a state_dict checkpoint.
    batch_size : number of temporal windows per inference batch.

    Returns
    -------
    Path to the output poses_3d.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kpts_2d_raw, keypoint_names_raw, frame_ids = load_2d_poses(poses_2d_path)
    kpts_2d, keypoint_names, did_convert = convert_keypoints_to_h36m_17(
        kpts_2d_raw,
        keypoint_names_raw,
    )
    hip_indices = get_hip_indices(keypoint_names)
    n_frames = kpts_2d.shape[0]

    if did_convert:
        print("Converted input keypoints from COCO-17 to H36M-17 for MotionBERT consistency.")
    _write_2d_h36m_manifest(output_dir, frame_ids, kpts_2d)

    print(f"Lifting {n_frames} frames to 3D (model={model_name}, window={receptive_field})")

    if model_name == "motionbert":
        poses_3d = _lift_motionbert(
            kpts_2d,
            checkpoint_path=checkpoint_path,
            checkpoint_url=checkpoint_url,
            output_dir=output_dir,
            device=device,
            receptive_field=receptive_field,
            model_kwargs=model_kwargs,
            batch_size=batch_size,
            hip_indices=hip_indices,
        )
    else:
        print(f"Warning: unknown model '{model_name}', using naive baseline lift.")
        poses_3d = _lift_naive(kpts_2d, hip_indices=hip_indices)

    # Save results
    out_path = output_dir / "poses_3d.json"
    result = {
        "model": model_name,
        "n_frames": int(poses_3d.shape[0]),
        "keypoint_names": keypoint_names,
        "frames": [
            {
                "frame_id": int(frame_ids[i]),
                "keypoints_3d": poses_3d[i].tolist(),  # (17, 3)
            }
            for i in range(poses_3d.shape[0])
        ],
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  3D poses saved to {out_path}")
    return out_path


def _load_keypoint_names(poses_2d_path: str | Path) -> list[str]:
    with open(poses_2d_path) as f:
        data = json.load(f)
    return data.get("keypoint_names", [])


def _write_2d_h36m_manifest(
    output_dir: Path,
    frame_ids: list[int],
    keypoints_h36m: np.ndarray,
) -> None:
    """Write converted 2D keypoints in H36M order for debugging and audits."""
    out_path = output_dir / "poses_2d_h36m.json"
    payload = {
        "keypoint_names": H36M_KEYPOINTS_17,
        "n_frames": int(keypoints_h36m.shape[0]),
        "frames": [
            {
                "frame_id": int(frame_ids[i]),
                "keypoints": keypoints_h36m[i].tolist(),
            }
            for i in range(keypoints_h36m.shape[0])
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def _lift_motionbert(
    kpts_2d: np.ndarray,
    *,
    checkpoint_path: str | Path | None,
    checkpoint_url: str | None,
    output_dir: Path,
    device: str = "mps",
    receptive_field: int = 243,
    model_kwargs: dict[str, Any] | None = None,
    batch_size: int = 16,
    hip_indices: tuple[int, int] = (4, 1),
) -> np.ndarray:
    """Lift 2D->3D using MotionBERT.

    Supports two checkpoint styles:
    1) TorchScript model (.pt/.jit) loaded with ``torch.jit.load``.
    2) State dict checkpoints that require ``motionbert.model.DSTformer``.
    """
    try:
        __import__("torch")
    except ImportError as exc:
        raise ImportError("torch is required for MotionBERT lifting") from exc

    n_frames, _, _ = kpts_2d.shape

    ckpt_path = _resolve_checkpoint_path(checkpoint_path, checkpoint_url, output_dir)
    model = _load_motionbert_model(
        ckpt_path,
        device=device,
        model_kwargs=model_kwargs,
    )

    # Normalize 2D keypoints to a person-centered coordinate system.
    xy = kpts_2d[:, :, :2].copy()
    confidence = kpts_2d[:, :, 2:3]

    # Center on hip midpoint in the active keypoint convention.
    left_hip_idx, right_hip_idx = hip_indices
    hip_center = (xy[:, left_hip_idx:left_hip_idx + 1, :] + xy[:, right_hip_idx:right_hip_idx + 1, :]) / 2.0
    xy = xy - hip_center

    # Scale so max extent is approximately 1.
    scale = np.abs(xy).max(axis=(1, 2), keepdims=True) + 1e-6
    xy = xy / scale

    poses_3d = _infer_motionbert_windows(
        model=model,
        xy=xy,
        confidence=confidence,
        device=device,
        receptive_field=receptive_field,
        batch_size=batch_size,
    )

    if poses_3d.shape != (n_frames, 17, 3):
        raise RuntimeError(
            "MotionBERT output shape mismatch. "
            f"Expected {(n_frames, 17, 3)}, got {poses_3d.shape}."
        )

    return poses_3d


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoint_url: str | None,
    output_dir: Path,
) -> Path:
    """Resolve checkpoint path, optionally downloading it when URL is provided."""
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"MotionBERT checkpoint not found: {path}. "
                "Set pose_3d.checkpoint to a valid file."
            )
        return path

    if checkpoint_url:
        cache_dir = output_dir.parent / "_model_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = checkpoint_url.rstrip("/").split("/")[-1] or "motionbert_checkpoint.pt"
        if "?" in filename:
            filename = filename.split("?", 1)[0]
        target = cache_dir / filename
        if not target.exists():
            print(f"  Downloading MotionBERT checkpoint from {checkpoint_url}")
            urlretrieve(checkpoint_url, target)
        return target.resolve()

    raise FileNotFoundError(
        "No MotionBERT checkpoint configured. "
        "Set pose_3d.checkpoint to a local file or pose_3d.checkpoint_url to download it."
    )


def _load_motionbert_model(
    checkpoint_path: Path,
    *,
    device: str,
    model_kwargs: dict[str, Any] | None,
) -> Any:
    """Load MotionBERT model from checkpoint as TorchScript or DSTformer state dict."""
    import torch

    # First preference: TorchScript checkpoint, easiest for deployment.
    try:
        scripted = torch.jit.load(str(checkpoint_path), map_location=device)
        scripted.eval()
        return scripted
    except Exception:
        pass

    checkpoint_obj = torch.load(str(checkpoint_path), map_location=device)
    state_dict = _extract_state_dict(checkpoint_obj)
    if state_dict is None:
        raise RuntimeError(
            "Unsupported checkpoint format. "
            "Expected TorchScript model or a state_dict-style checkpoint."
        )

    try:
        from motionbert.model import DSTformer  # type: ignore
    except ImportError:
        DSTformer = _load_dstformer_from_official_source(checkpoint_path.parent)

    if not model_kwargs:
        raise RuntimeError(
            "pose_3d.model_kwargs is required for state-dict checkpoints so DSTformer can be instantiated."
        )

    model = DSTformer(**model_kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: missing keys when loading MotionBERT: {len(missing)}")
    if unexpected:
        print(f"  Warning: unexpected keys when loading MotionBERT: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model


def _load_dstformer_from_official_source(cache_root: Path) -> Any:
    """Load DSTformer class from official MotionBERT source as runtime fallback."""
    source_root = cache_root / "motionbert_src"
    dstformer_path = source_root / "lib" / "model" / "DSTformer.py"
    drop_path = source_root / "lib" / "model" / "drop.py"

    if not dstformer_path.exists() or not drop_path.exists():
        (source_root / "lib" / "model").mkdir(parents=True, exist_ok=True)
        base = "https://huggingface.co/walterzhu/MotionBERT/resolve/main"
        urlretrieve(f"{base}/lib/model/DSTformer.py?download=true", dstformer_path)
        urlretrieve(f"{base}/lib/model/drop.py?download=true", drop_path)

        for pkg in (
            source_root / "lib",
            source_root / "lib" / "model",
        ):
            init_file = pkg / "__init__.py"
            if not init_file.exists():
                init_file.write_text("", encoding="utf-8")

    source_path_str = str(source_root)
    if source_path_str not in sys.path:
        sys.path.insert(0, source_path_str)

    from lib.model.DSTformer import DSTformer  # type: ignore

    return DSTformer


def _extract_state_dict(checkpoint_obj: Any) -> dict[str, Any] | None:
    """Extract a model state dict from common checkpoint layouts."""
    if isinstance(checkpoint_obj, dict):
        if all(isinstance(k, str) for k in checkpoint_obj.keys()):
            if any(k.startswith("module.") or "." in k for k in checkpoint_obj.keys()):
                return _strip_prefixes(checkpoint_obj)

        for key in ("state_dict", "model_state", "model", "model_pos"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return _strip_prefixes(value)
    return None


def _strip_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip common wrapper prefixes from checkpoint parameter keys."""
    prefixes = ("module.", "model.", "backbone.")
    keys = list(state_dict.keys())
    for prefix in prefixes:
        if keys and all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    if any(k.startswith("module.") for k in keys):
        return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    return state_dict


def _infer_motionbert_windows(
    model: Any,
    *,
    xy: np.ndarray,
    confidence: np.ndarray,
    device: str,
    receptive_field: int,
    batch_size: int,
) -> np.ndarray:
    """Run windowed temporal inference and return (N, 17, 3)."""
    import torch

    n_frames = xy.shape[0]
    receptive_field = max(1, int(receptive_field))
    if receptive_field % 2 == 0:
        receptive_field += 1
    half = receptive_field // 2

    # Build per-frame temporal windows with edge padding.
    windows_xy: list[np.ndarray] = []
    windows_xyc: list[np.ndarray] = []
    for frame_idx in range(n_frames):
        start = max(0, frame_idx - half)
        end = min(n_frames, frame_idx + half + 1)
        window_xy = xy[start:end]
        window_xyc = np.concatenate([xy[start:end], confidence[start:end]], axis=-1)

        pad_left = max(0, half - frame_idx)
        pad_right = max(0, frame_idx + half + 1 - n_frames)
        if pad_left or pad_right:
            window_xy = np.pad(
                window_xy,
                ((pad_left, pad_right), (0, 0), (0, 0)),
                mode="edge",
            )
            window_xyc = np.pad(
                window_xyc,
                ((pad_left, pad_right), (0, 0), (0, 0)),
                mode="edge",
            )

        windows_xy.append(window_xy)
        windows_xyc.append(window_xyc)

    x2 = torch.from_numpy(np.stack(windows_xy, axis=0)).float().to(device)
    x3 = torch.from_numpy(np.stack(windows_xyc, axis=0)).float().to(device)
    expected_dim = _infer_model_input_dim(model)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for idx in range(0, n_frames, batch_size):
            batch2 = x2[idx: idx + batch_size]
            batch3 = x3[idx: idx + batch_size]
            if expected_dim == 2:
                out = model(batch2)
            else:
                out = model(batch3)
            center = _select_center_predictions(out, center_index=half)
            preds.append(center)

    return np.concatenate(preds, axis=0).astype(np.float32)


def _infer_model_input_dim(model: Any) -> int:
    """Infer expected per-joint input dim from model metadata when available."""
    joints_embed = getattr(model, "joints_embed", None)
    in_features = getattr(joints_embed, "in_features", None)
    if isinstance(in_features, int) and in_features in (2, 3):
        return in_features
    return 3


def _select_center_predictions(output: Any, *, center_index: int) -> np.ndarray:
    """Normalize model outputs and return center-frame prediction as numpy array."""
    import torch

    if isinstance(output, dict):
        for key in ("pred_3d", "output", "pred", "poses_3d"):
            if key in output:
                output = output[key]
                break
    elif isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise RuntimeError("MotionBERT output is not a torch.Tensor after normalization.")

    # Expected forms:
    # - (B, T, 17, 3): take center frame
    # - (B, 17, 3): already per-sample prediction
    if output.ndim == 4:
        if output.shape[-2:] != (17, 3):
            raise RuntimeError(f"Unexpected MotionBERT output shape: {tuple(output.shape)}")
        output = output[:, center_index, :, :]
    elif output.ndim == 3:
        if output.shape[-2:] != (17, 3):
            raise RuntimeError(f"Unexpected MotionBERT output shape: {tuple(output.shape)}")
    else:
        raise RuntimeError(f"Unsupported MotionBERT output rank: {output.ndim}")

    return output.detach().cpu().numpy()


def _lift_naive(
    kpts_2d: np.ndarray,
    *,
    hip_indices: tuple[int, int] = (4, 1),
) -> np.ndarray:
    """Baseline: estimate Z from bone lengths heuristic.

    This is a simple placeholder — it creates a plausible-looking 3D skeleton
    by assigning depth proportional to vertical position (lower = closer).
    It is NOT accurate but lets you test the full pipeline.
    """
    n_frames, n_joints, _ = kpts_2d.shape
    xy = kpts_2d[:, :, :2].copy()

    # Centre on hip midpoint
    left_hip_idx, right_hip_idx = hip_indices
    hip_center = (xy[:, left_hip_idx:left_hip_idx + 1, :] + xy[:, right_hip_idx:right_hip_idx + 1, :]) / 2.0
    xy = xy - hip_center

    # Normalise
    scale = np.abs(xy).max(axis=(1, 2), keepdims=True) + 1e-6
    xy = xy / scale

    # Fake Z: use a simple heuristic
    z = np.zeros((n_frames, n_joints, 1), dtype=np.float32)
    z[:, :, 0] = -xy[:, :, 1] * 0.3  # higher points slightly further away

    poses_3d = np.concatenate([xy, z], axis=-1)  # (N, 17, 3)
    return poses_3d
