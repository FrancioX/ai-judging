"""Batch-process all usable Snowboard Men videos through the full pipeline.

Runs frames → segmentation → tracking → pose_2d → pose_3d for each of the 22
usable Snowboard Men videos (excluding the 0-score video). Skips stages whose
outputs already exist to avoid recomputation.

Usage
-----
    uv run python scripts/batch_snowboard_pipeline.py
    uv run python scripts/batch_snowboard_pipeline.py --force         # recompute all
    uv run python scripts/batch_snowboard_pipeline.py --stage pose_3d # only missing pose_3d
    uv run python scripts/batch_snowboard_pipeline.py --continue-on-error
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
RAW_VIDEOS_DIR = WORKSPACE / "raw_videos"
OUTPUT_DIR = WORKSPACE / "output"

# 22 usable Snowboard Men stems (excluding 0-score Martxelo Urruzola)
SNOWBOARD_MEN_STEMS = [
    "Snowboard Men_1_80_Fabian Knözinger",
    "Snowboard Men_2_71.67_Aleksandr Girnik",
    "Snowboard Men_3_69.33_Jakob Weger",
    "Snowboard Men_4_67.33_Pol Sabidó Juvé",
    "Snowboard Men_5_64_Tom Leclercq Chabenat",
    "Snowboard Men_6_63_Tristan Legru-Yildiz",
    "Snowboard Men_7_61.67_Soini Turunen",
    "Snowboard Men_8_56_Lucas Muñoz",
    "Snowboard Men_9_55_Matthew Podelnyk",
    "Snowboard Men_10_54.33_Cedric Giraudeau",
    "Snowboard Men_11_53_Nicolas Lagger",
    "Snowboard Men_12_52.33_Adriano Cardillo",
    "Snowboard Men_13_51.67_Alexandre Fournier-Simu",
    "Snowboard Men_14_48.67_Taketo Kinoshita",
    "Snowboard Men_15_45_Theodor Salen",
    "Snowboard Men_16_40_Jonatan Laland",
    "Snowboard Men_17_35_Quentin Puydenus",
    "Snowboard Men_18_31_Tom Hill",
    "Snowboard Men_19_19_Thomas Leisi",
    "Snowboard Men_20_15_Oscar Weatherall",
    "Snowboard Men_21_12_Daniel Nufer",
    "Snowboard Men_22_8.67_Gregory Whitehead",
]

ALL_STAGES = ["frames", "segmentation", "tracking", "pose_2d", "pose_3d"]

# Output sentinel files per stage (relative to output/<stage>/<stem>/)
STAGE_SENTINELS: dict[str, str] = {
    "frames": "frames_meta.json",
    "segmentation": "segmentation.json",
    "tracking": "tracking.json",
    "pose_2d": "poses_2d.json",
    "pose_3d": "poses_3d.json",
}


def _stage_done(stem: str, stage: str) -> bool:
    sentinel = STAGE_SENTINELS[stage]
    stage_dir = {
        "frames": "frames",
        "segmentation": "segmentation",
        "tracking": "tracking",
        "pose_2d": "poses_2d",
        "pose_3d": "poses_3d",
    }[stage]
    path = OUTPUT_DIR / stage_dir / stem / sentinel
    return path.exists()


def _run_stage(video_path: Path, stage: str) -> float:
    """Run a single stage and return elapsed seconds."""
    cmd = [
        "uv", "run", "python", "-m", "src.pipeline",
        str(video_path),
        "--stage", stage,
    ]
    print(f"\n  [{stage}] {video_path.name}")
    t0 = time.monotonic()
    subprocess.run(cmd, check=True, cwd=WORKSPACE)
    return time.monotonic() - t0


def process_video(
    stem: str,
    *,
    stages: list[str],
    force: bool,
    continue_on_error: bool,
) -> dict[str, float]:
    """Process a single video. Returns dict of stage → elapsed seconds (skipped = -1)."""
    video_path = RAW_VIDEOS_DIR / f"{stem}.mp4"
    timings: dict[str, float] = {}

    if not video_path.exists():
        print(f"  WARNING: video not found: {video_path}")
        return timings

    for stage in stages:
        if not force and _stage_done(stem, stage):
            print(f"  [skip] {stage} already done")
            timings[stage] = -1.0
            continue
        try:
            elapsed = _run_stage(video_path, stage)
            timings[stage] = elapsed
            print(f"  [{stage}] done in {elapsed:.0f}s")
        except subprocess.CalledProcessError as exc:
            print(f"  ERROR: stage={stage} failed: {exc}")
            timings[stage] = float("nan")
            if not continue_on_error:
                raise
            break
    return timings


def _fmt_seconds(s: float) -> str:
    if s < 0:
        return "skip"
    if s != s:  # nan
        return "ERROR"
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s" if m else f"{sec}s"


def _print_timing_table(
    timing_records: list[tuple[str, dict[str, float]]],
    stages: list[str],
) -> None:
    col_w = 10
    stem_w = 40
    header = f"{'Video':<{stem_w}}" + "".join(f"  {s:>{col_w}}" for s in stages) + f"  {'TOTAL':>{col_w}}"
    print(f"\n{'='*len(header)}")
    print("  Timing Summary")
    print(f"{'='*len(header)}")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    stage_totals: dict[str, float] = {s: 0.0 for s in stages}
    stage_counts: dict[str, int] = {s: 0 for s in stages}

    for stem, timings in timing_records:
        row_total = sum(v for v in timings.values() if v > 0 and v == v)
        row = f"  {stem[:stem_w]:<{stem_w}}"
        for s in stages:
            v = timings.get(s, float("nan"))
            row += f"  {_fmt_seconds(v):>{col_w}}"
            if v > 0 and v == v:
                stage_totals[s] += v
                stage_counts[s] += 1
        row += f"  {_fmt_seconds(row_total):>{col_w}}"
        print(row)

    print(f"  {'-'*len(header)}")
    avg_row = f"  {'AVERAGE':<{stem_w}}"
    for s in stages:
        avg = stage_totals[s] / stage_counts[s] if stage_counts[s] else 0.0
        avg_row += f"  {_fmt_seconds(avg):>{col_w}}"
    print(avg_row)
    print(f"{'='*len(header)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-process Snowboard Men videos")
    parser.add_argument(
        "--stage",
        choices=ALL_STAGES,
        default=None,
        help="Only run this single stage (and any missing upstream stages). "
             "Default: run all stages from frames to pose_3d.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all stages even if outputs already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with other videos when a stage fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which stages would run without executing them.",
    )
    args = parser.parse_args()

    stages = ALL_STAGES if args.stage is None else ALL_STAGES[: ALL_STAGES.index(args.stage) + 1]

    print(f"Processing {len(SNOWBOARD_MEN_STEMS)} Snowboard Men videos")
    print(f"Stages: {' → '.join(stages)}")
    print(f"Force: {args.force}")
    print()

    failures: list[str] = []
    timing_records: list[tuple[str, dict[str, float]]] = []

    for stem in SNOWBOARD_MEN_STEMS:
        print(f"\n{'='*60}")
        print(f"  {stem}")
        print(f"{'='*60}")

        if args.dry_run:
            for stage in stages:
                done = _stage_done(stem, stage)
                status = "done" if done and not args.force else "WILL RUN"
                print(f"  [{status}] {stage}")
            continue

        try:
            timings = process_video(
                stem,
                stages=stages,
                force=args.force,
                continue_on_error=args.continue_on_error,
            )
            timing_records.append((stem, timings))
            had_error = any(v != v for v in timings.values())  # nan check
            if had_error:
                failures.append(stem)
        except Exception as exc:
            failures.append(stem)
            timing_records.append((stem, {}))
            print(f"FATAL: {stem}: {exc}")
            if not args.continue_on_error:
                sys.exit(1)

    if timing_records and not args.dry_run:
        _print_timing_table(timing_records, stages)

    print(f"\n{'='*60}")
    if failures:
        print(f"Completed with {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        n = len([r for r in timing_records if r])
        print(f"All {n} videos processed successfully.")


if __name__ == "__main__":
    main()
