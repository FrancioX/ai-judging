#!/usr/bin/env python3
"""Show which track_id the continuity resolver actually follows in 300-600 range."""
import json
import math

with open('output/tracking/VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86/tracking.json') as f:
    track_data = json.load(f)

with open('output/segmentation/VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86/segmentation.json') as f:
    seg_data = json.load(f)

# For each tracking frame 300-600, find which segmentation track_id matches
# the tracking bbox center
for frame_t in track_data['frames']:
    fid = frame_t['frame_id']
    if not (300 <= fid <= 600) or fid % 5 != 0:
        continue

    t_bbox = frame_t['bbox']
    t_cx = (t_bbox[0] + t_bbox[2]) / 2
    t_cy = (t_bbox[1] + t_bbox[3]) / 2

    # Find closest segmentation person
    seg_frame = seg_data['frames'][fid]
    best_dist = 9999
    best_tid = "?"
    all_persons = ""

    for p in seg_frame.get('persons', []):
        bbox = p['bbox']
        p_cx = (bbox[0] + bbox[2]) / 2
        p_cy = (bbox[1] + bbox[3]) / 2
        dist = math.sqrt((t_cx - p_cx)**2 + (t_cy - p_cy)**2)
        all_persons += f"  T{p.get('track_id','?')}=({p_cx:.0f},{p_cy:.0f})"
        if dist < best_dist:
            best_dist = dist
            best_tid = p.get('track_id', '?')

    det = "det" if frame_t['detected'] else "int"
    print(f"F{fid:4d} [{det}] tracking=({t_cx:.0f},{t_cy:.0f}) → follows Track {best_tid} (dist={best_dist:.0f})  |  available:{all_persons}")
