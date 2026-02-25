#!/usr/bin/env python3
"""Analyze interpolation patterns in Lach Powell tracking."""
import json

# Read tracking output
with open('output/tracking/VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86/tracking.json') as f:
    track_data = json.load(f)

# Read segmentation for comparison
with open('output/segmentation/VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86/segmentation.json') as f:
    seg_data = json.load(f)

# Analyze interpolated frames and their distribution
interpolated_frames = []
detected_frames = []

for frame in track_data['frames']:
    if frame['detected']:
        detected_frames.append(frame['frame_id'])
    elif frame['interpolated']:
        interpolated_frames.append(frame['frame_id'])

print(f"Detected frames: {len(detected_frames)}")
print(f"Interpolated frames: {len(interpolated_frames)}")
print(f"Total frames: {len(track_data['frames'])}")
print()

# Find longest interpolation stretch
if interpolated_frames:
    stretches = []
    current_stretch = [interpolated_frames[0]]
    for i in range(1, len(interpolated_frames)):
        if interpolated_frames[i] == interpolated_frames[i-1] + 1:
            current_stretch.append(interpolated_frames[i])
        else:
            stretches.append(current_stretch)
            current_stretch = [interpolated_frames[i]]
    stretches.append(current_stretch)

    stretches.sort(key=len, reverse=True)

    print(f"Top 10 longest interpolation stretches:")
    print(f"{'Start':<8} {'End':<8} {'Length':<10} {'Seg detections':<15}")
    print("-" * 50)
    for stretch in stretches[:10]:
        start_frame = min(stretch)
        end_frame = max(stretch)
        length = len(stretch)

        # Check if segmentation has detections in this range
        seg_det_count = 0
        for frame in seg_data['frames']:
            if start_frame <= frame['frame_id'] <= end_frame:
                if frame.get('detected'):
                    seg_det_count += 1

        print(f"{start_frame:<8} {end_frame:<8} {length:<10} {seg_det_count:<15}")

# Deep dive into the largest interpolation stretch
if stretches:
    print()
    print(f"Details of largest interpolation stretch (frames {min(stretches[0])}-{max(stretches[0])}):")
    largest_start = min(stretches[0])
    largest_end = max(stretches[0])

    # Check what detections exist in segmentation during this period
    print(f"\nSegmentation detections in this range:")
    print(f"{'Frame':<8} {'Detected':<10} {'Track IDs':<20}")
    print("-" * 40)

    for frame in seg_data['frames']:
        if largest_start <= frame['frame_id'] <= largest_end:
            track_ids = []
            if 'persons' in frame and frame['persons']:
                for person in frame['persons']:
                    track_id = person.get('track_id')
                    if track_id is not None:
                        track_ids.append(str(track_id))

            detected_str = "Yes" if frame.get('detected') else "No"
            track_str = ", ".join(track_ids) if track_ids else "none"
            print(f"{frame['frame_id']:<8} {detected_str:<10} {track_str:<20}")
