from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MINI_SET_STEMS = [
    "VERBIER FREERIDE WEEK QUALIFIER 4__2_Ski Men_Andreas Bakke_24_Norway_89",
    "VERBIER FREERIDE WEEK QUALIFIER 4__1_Ski Men_Arno Vuarnier_58_Switzerland_91.33",
    "VERBIER FREERIDE WEEK QUALIFIER 4__16_Snowboard Men_Jonatan Laland_61_Norway_40_",
    "VERBIER FREERIDE WEEK QUALIFIER 4__17_Snowboard Men_Quentin Puydenus_53_France_35_",
]

DEFAULT_STAGES = ["tracking", "pose_2d", "pose_3d", "visualization"]


def _resolve_video(stem: str, roots: list[Path]) -> Path:
    for root in roots:
        candidate = root / f"{stem}.mp4"
        if candidate.exists():
            return candidate
    root_list = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(f"Could not find {stem}.mp4 in: {root_list}")


def _run_stage(video_path: Path, stage: str) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.pipeline",
        str(video_path),
        "--stage",
        stage,
    ]
    print(f"\n=== {stage}: {video_path.name} ===")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mini dev set for pose3d and visualization")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=DEFAULT_STAGES,
        choices=["tracking", "pose_2d", "pose_3d", "visualization"],
        help="Stages to run for each mini-set video",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other videos when a stage fails",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    roots = [workspace / "raw_videos", workspace / "raw_videos_snowboard"]

    failures: list[str] = []
    for stem in MINI_SET_STEMS:
        try:
            video_path = _resolve_video(stem, roots)
            for stage in args.stages:
                _run_stage(video_path, stage)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{stem}: {exc}")
            print(f"ERROR: {stem}: {exc}")
            if not args.continue_on_error:
                break

    if failures:
        print("\nMini dev run completed with failures:")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print("\nMini dev run completed successfully.")


if __name__ == "__main__":
    main()
