import json
import os


def bbox_iou(boxA, boxB):
    """Compute IoU between two bounding boxes."""
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter_area = interW * interH

    areaA = (boxA["x2"] - boxA["x1"]) * (boxA["y2"] - boxA["y1"])
    areaB = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def convert_to_frame_dict(entries, key):
    """
    Convert dataset structure:
    [
        { "video_id": "...", key: [ { "bboxes": [ ... ] }, ... ] },
        ...
    ]

    into:
        { video_id: { frame: bbox_dict } }
    """
    out = {}
    for item in entries:
        vid = item["video_id"]
        frame_dict = {}
        for group in item.get(key, []):
            for b in group.get("bboxes", []):
                frame_dict[b["frame"]] = b
        out[vid] = frame_dict
    return out


def compute_st_iou(groundtruth_path, submission_path):
    """Compute ST-IoU given json file paths."""

    # --- Load JSON files ---
    with open(groundtruth_path, "r") as f:
        gt = json.load(f)

    with open(submission_path, "r") as f:
        sub = json.load(f)

    # Convert into {video_id: {frame: bbox}}
    gt_dict = convert_to_frame_dict(gt, "annotations")
    sub_dict = convert_to_frame_dict(sub, "detections")

    st_iou_results = {}

    for vid, gt_frames in gt_dict.items():

        sub_frames = sub_dict.get(vid, {})
        common_frames = set(gt_frames.keys()) & set(sub_frames.keys())

        if not common_frames:
            st_iou_results[vid] = 0.0
            continue

        ious = [
            bbox_iou(gt_frames[f], sub_frames[f])
            for f in common_frames
        ]

        st_iou_results[vid] = sum(ious) / len(ious)

    return st_iou_results
