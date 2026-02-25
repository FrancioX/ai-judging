#!/usr/bin/env python3
"""Analyze which track was selected and why."""
import json

# Read tracking output
with open('output/tracking/VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86/tracking.json') as f:
    track_data = json.load(f)

print(f"Selected track ID: {track_data['selected_track_id']}")
print(f"Track score: {track_data['track_score']:.4f}")
print(f"Merged tracks: {track_data['merged_tracks']}")
print(f"Weights: {track_data['weights']}")
print()

# Check detection pattern of selected track vs interpolations
selected_track = track_data['selected_track_id']

# Look at the problem zone (521-883)
print("Analysis of frames 521-883 in tracking output:")
print(f"{'Frame':<8} {'Detected':<12} {'Interpolated':<15} {'Confidence':<12} {'Track ID':<10}")
print("-" * 60)

problem_detected = 0
problem_interpolated = 0

for frame in track_data['frames']:
    if 521 <= frame['frame_id'] <= 883:
        if frame['frame_id'] in [521, 539, 545, 586, 650, 750, 850, 880]:  # Sample frames
            det_str = "Yes" if frame['detected'] else "No"
            interp_str = "Yes" if frame['interpolated'] else "No"
            conf = frame.get('confidence', 0)
            track_id = frame.get('track_id', -1)
            print(f"{frame['frame_id']:<8} {det_str:<12} {interp_str:<15} {conf:<12.4f} {track_id:<10}")

        if frame['detected']:
            problem_detected += 1
        if frame['interpolated']:
            problem_interpolated += 1

print()
print(f"Summary for zone 521-883:")
print(f"  Detected in tracking: {problem_detected}")
print(f"  Interpolated in tracking: {problem_interpolated}")
