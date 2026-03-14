from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from src.pose_3d.lifter import lift_to_3d


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    p3d_cfg = config.get("pose_3d", {})

    video_stem = "Ski Men_2_89_Andreas Bakke"
    poses_2d_path = Path(f"output/poses_2d/{video_stem}/poses_2d.json")
    poses_2d_50_path = Path(f"output/poses_2d/{video_stem}/poses_2d_50frames.json")
    out_dir = Path(f"output/poses_3d/{video_stem}_50frames")

    with open(poses_2d_path) as f:
        data = json.load(f)
    frames = data["frames"][:50]
    data["frames"] = frames
    data["n_frames"] = len(frames)
    with open(poses_2d_50_path, "w") as f:
        json.dump(data, f, indent=2)

    start = time.perf_counter()
    out_path = lift_to_3d(
        poses_2d_50_path,
        out_dir,
        model_name="motionbert",
        checkpoint_path="output/poses_3d/_model_cache/best_epoch.bin",
        checkpoint_url=None,
        device=p3d_cfg.get("device", "mps"),
        receptive_field=p3d_cfg.get("receptive_field", 243),
        model_kwargs={
            "dim_in": 3,
            "dim_out": 3,
            "dim_feat": 512,
            "dim_rep": 512,
            "depth": 5,
            "num_heads": 8,
            "mlp_ratio": 2,
            "num_joints": 17,
            "maxlen": 243,
            "att_fuse": True,
        },
        batch_size=p3d_cfg.get("batch_size", 16),
    )
    elapsed = time.perf_counter() - start
    fps = len(frames) / elapsed if elapsed > 0 else 0

    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"effective_fps={fps:.3f}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
