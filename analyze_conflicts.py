#!/usr/bin/env python3
"""Analyze tracking conflicts in frames 300-600 (10-20s at 30fps) for Lach Powell."""
import json
import math

with open('output/tracking/VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86/tracking.json') as f:
    track_data = json.load(f)

with open('output/segmentation/VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86/segmentation.json') as f:
    seg_data = json.load(f)

# 10-20 seconds at 30fps = frames 300-600
print("Tracking output for frames 300-600 (10-20s):")
print(f"{'Frame':<8} {'Detected':<10} {'Track ID':<10} {'Conf':<8} {'BBox center':<20}")
print("-" * 60)

prev_cx, prev_cy = None, None
for frame in track_data['frames']:
    if 300 <= frame['frame_id'] <= 600:
        bbox = frame['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        # Compute jump from previous frame
        jump = ""
        if prev_cx is not None:
            dist = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            if dist > 30:
                jump = f" ← JUMP {dist:.0f}px"

        det = "det" if frame['detected'] else "interp"
        tid = frame.get('track_id', -1)

        if frame['frame_id'] % 10 == 0 or jump:  # Print every 10th frame + jumps
            print(f"{frame['frame_id']:<8} {det:<10} {tid:<10} {frame['confidence']:<8.3f} ({cx:.0f}, {cy:.0f}){jump}")

        prev_cx, prev_cy = cx, cy

# Also show the segmentation persons at conflict frames
print("\n\nSegmentation detections at key frames (300-600):")
for frame in seg_data['frames']:
    if frame['frame_id'] in [300, 310, 320, 330, 340, 350, 400, 450, 500, 520, 530]:
        persons = frame.get('persons', [])
        print(f"\nFrame {frame['frame_id']}: {len(persons)} persons")
        for p in persons:
            bbox = p['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            print(f"  Track {p.get('track_id', '?'):>3}: conf={p['confidence']:.3f} center=({cx:.0f},{cy:.0f}) area={p['area']}")
