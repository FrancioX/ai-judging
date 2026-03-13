---
name: run-tracking
description: 'Run the tracking stage for one or many videos, including timed runs for experiment benchmarking. Use when you need to regenerate tracking outputs quickly after config or tracker changes.'
argument-hint: 'Provide video path(s), stage options, and whether to run timed mode.'
user-invocable: true
---

# Run Tracking

## When To Use
- Re-run tracking after config or code changes.
- Benchmark tracking runtime across experiments.
- Regenerate outputs for a single video or a full annotated batch.

## Procedure
1. Confirm tracking config in `config.yaml`.
2. Run timed tracking for a single video:

```bash
/usr/bin/time -p uv run python -m src.pipeline "raw_videos/VIDEO.mp4" --stage tracking
```

3. Run timed tracking for annotated batch (example loop by GT stems):

```bash
/usr/bin/time -p bash -lc '
set -euo pipefail
for gt in annotations/tracking/*/gt_centers.csv; do
  stem="$(basename "$(dirname "$gt")")"
  video=""
  for base in raw_videos raw_videos_snowboard; do
    cand="$base/${stem}.mp4"
    if [[ -f "$cand" ]]; then
      video="$cand"
      break
    fi
  done
  if [[ -z "$video" ]]; then
    echo "Missing video for stem: $stem" >&2
    continue
  fi
  echo "[tracking] $video"
  uv run python -m src.pipeline "$video" --stage tracking
 done
'
```

4. Save runtime and command used in experiment notes.

## Notes
- `--stage tracking` reuses segmentation outputs and only runs temporal tracking.
- If a dependency is missing, the stage fails with a clear error.
