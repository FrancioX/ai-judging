#!/usr/bin/env python3
"""Calculate track scores for Lach Powell to see why 13,14,16,19 are excluded."""
import json
import math
from collections import Counter, defaultdict

# Read segmentation data
with open('output/segmentation/VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86/segmentation.json') as f:
    seg_data = json.load(f)

# Re-compute track scores
tracks = defaultdict(list)
img_w, img_h = seg_data['frames'][0].get('frame_width', 1920), seg_data['frames'][0].get('frame_height', 1080)

# If not in data, estimate from bbox observations
for frame in seg_data['frames']:
    if 'persons' in frame and frame['persons']:
        for p in frame['persons']:
            tid = p.get('track_id')
            if tid is None:
                continue
            tracks[tid].append({
                'frame_id': frame['frame_id'],
                'bbox': p['bbox'],
                'confidence': p['confidence'],
                'area': p['area'],
            })

n_frames = len(seg_data['frames'])
max_dist = math.sqrt((img_w / 2) ** 2 + (img_h / 2) ** 2)
cx, cy = img_w / 2, img_h / 2

# Config params (from config.yaml)
w_conf = 0.3
w_center = 0.5
w_length = 0.2
min_track_frames = 10
merge_top_n = 5
merge_score_threshold = 0.3

# Score all tracks
track_scores = []

for tid, obs in sorted(tracks.items()):
    if len(obs) < min_track_frames:
        continue

    mean_conf = sum(o['confidence'] for o in obs) / len(obs)

    # Center proximity
    center_scores = []
    for o in obs:
        bx = (o['bbox'][0] + o['bbox'][2]) / 2
        by = (o['bbox'][1] + o['bbox'][3]) / 2
        dist = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
        center_scores.append(1.0 - dist / max_dist)
    mean_center = sum(center_scores) / len(center_scores)

    length_ratio = len(obs) / n_frames

    score = (w_conf * mean_conf +
             w_center * mean_center +
             w_length * length_ratio)

    track_scores.append((tid, score, len(obs)))

# Sort by score
track_scores.sort(key=lambda x: x[1], reverse=True)

print(f"All tracks ranked by score:")
print(f"{'Rank':<6} {'Track ID':<10} {'Score':<10} {'Detections':<12} {'Selected?':<12}")
print("-" * 55)

for rank, (tid, score, n_det) in enumerate(track_scores, 1):
    selected = "✓ YES" if rank <= merge_top_n and score >= merge_score_threshold else ""
    print(f"{rank:<6} {tid:<10} {score:<10.4f} {n_det:<12} {selected:<12}")

print()
print(f"Current config: merge_top_n={merge_top_n}, merge_score_threshold={merge_score_threshold}")
print(f"Selected tracks: {[tid for tid, _, _ in track_scores[:merge_top_n] if _ >= merge_score_threshold]}")
print()
print(f"Excluded but significant tracks:")
for tid, score, n_det in track_scores[merge_top_n:]:
    if n_det >= 10:  # Only show those meeting min_track_frames
        print(f"  Track {tid}: {n_det} frames (score={score:.4f})")
