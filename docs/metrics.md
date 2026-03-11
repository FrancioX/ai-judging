# Evaluation Metrics

## Detection (Segmentation)

| Metric | Description |
|---|---|
| **Detection Rate (Det%)** | % of GT frames where YOLO detected at least one person |
| **Mean Error (DetErr)** | Euclidean distance (px) between closest detected bbox center and GT center |
| **Pct Within Threshold (Det<Th)** | % of detected frames where closest bbox center is within threshold (default 50px) of GT |
| **Correct Person Rate (CorPer)** | In multi-person frames, % where ByteTrack's selected person is the one closest to GT |
| **Unique Track IDs (IDs)** | Number of distinct ByteTrack IDs assigned to the target athlete — measures fragmentation |
| **Longest Run Ratio (LRun)** | Longest consecutive sequence of the same track ID / total detected frames (1.0 = no fragmentation) |

## Tracking

| Metric | Description |
|---|---|
| **Mean Error (TrkErr)** | Euclidean distance (px) between tracked bbox center and GT center, after smoothing and gap-filling |
| **Pct Within Threshold (Trk<Th)** | % of GT frames where tracked center is within threshold (default 50px) |
| **HOTA** | Higher Order Tracking Accuracy — geometric mean of detection accuracy (DetA) and association accuracy (AssA), ranging 0–1 |
| **DetA** | Detection Accuracy component of HOTA — measures spatial overlap quality |
| **AssA** | Association Accuracy component of HOTA — measures identity preservation over time |

## Combined

| Metric | Description |
|---|---|
| **Improvement (Imprv)** | % reduction in mean error from detection to tracking. Positive = tracking helps, negative = tracking introduces error |

## Commands

```bash
# Combined detection + tracking table
uv run python -m src.evaluate

# Detection only
uv run python -m src.segmentation.evaluate --batch

# Tracking only
uv run python -m src.tracking.evaluate --batch
```
