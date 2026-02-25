#!/usr/bin/env python3
"""Analyze track lengths from segmentation data."""
import json
from collections import Counter, defaultdict

# Read segmentation data
with open('output/segmentation/VERBIER FREERIDE WEEK QUALIFIER 4__2_Ski Men_Andreas Bakke_24_Norway_89/segmentation.json') as f:
    seg_data = json.load(f)

# Count frames per track_id
track_lengths = Counter()
track_frames = defaultdict(list)

for frame in seg_data['frames']:
    if 'persons' in frame and frame['persons']:
        for person in frame['persons']:
            track_id = person.get('track_id')
            if track_id is not None and track_id >= 0:
                track_lengths[track_id] += 1
                track_frames[track_id].append(frame['frame_id'])

# Sort by track length
sorted_tracks = sorted(track_lengths.items(), key=lambda x: x[1], reverse=True)

print(f'Total tracks detected: {len(sorted_tracks)}')
print(f'Total frames in video: {len(seg_data["frames"])}')
print()

print('Top 20 longest tracks:')
print(f'{"Track ID":<10} {"Frames":<8} {"% video":<10} {"Frame range":<15}')
print('-' * 50)
for track_id, length in sorted_tracks[:20]:
    pct = (length / len(seg_data['frames'])) * 100
    frames = track_frames[track_id]
    frame_range = f'{min(frames)}-{max(frames)}'
    print(f'{track_id:<10} {length:<8} {pct:>6.2f}%   {frame_range:<15}')

print()
print('Impact of min_track_frames threshold:')
print(f'{"Threshold":<12} {"Ignored":<10} {"Kept":<8} {"Lost frames":<15} {"% lost":<10}')
print('-' * 60)
for threshold in [5, 10, 15, 20, 25, 30, 50]:
    ignored = sum(1 for _, l in sorted_tracks if l < threshold)
    kept = sum(1 for _, l in sorted_tracks if l >= threshold)
    lost_frames = sum(l for _, l in sorted_tracks if l < threshold)
    total_frames = sum(l for _, l in sorted_tracks)
    pct_lost = (lost_frames / total_frames) * 100 if total_frames > 0 else 0
    print(f'{threshold:<12} {ignored:<10} {kept:<8} {lost_frames:<15} {pct_lost:>7.2f}%')

print()
print(f'Short tracks (< current threshold of 10 frames):')
print(f'{"Track ID":<10} {"Frames":<8} {"Frame range":<15}')
print('-' * 40)
short_tracks = [(tid, l) for tid, l in sorted_tracks if l < 10]
for track_id, length in short_tracks:
    frames = track_frames[track_id]
    frame_range = f'{min(frames)}-{max(frames)}'
    print(f'{track_id:<10} {length:<8} {frame_range:<15}')

print()
print(f'Summary: {len(short_tracks)} tracks would be ignored with min_track_frames=10')
total_short = sum(l for tid, l in short_tracks)
print(f'These represent {total_short} detection frames ({(total_short/sum(l for _,l in sorted_tracks)*100):.2f}% of all detections)')
