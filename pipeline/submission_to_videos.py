import json
import cv2
import os

# ------------------------------------------------------
# Config
# ------------------------------------------------------
ANNOT_FILE = "data/output/submission_train.json"
VIDEO_DIR = "data/train/samples"
OUTPUT_DIR = "data/output/videos/train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------
# Load annotations
# ------------------------------------------------------
with open(ANNOT_FILE, "r") as f:
    annotations = json.load(f)

ann_by_video = {}
video_ids = []

for entry in annotations:
    vid = entry["video_id"]
    video_ids.append(vid)
    bboxes = {}

    for det in entry["detections"]:
        for box in det["bboxes"]:
            frame = box["frame"]
            if frame not in bboxes:
                bboxes[frame] = []
            bboxes[frame].append(box)

    ann_by_video[vid] = bboxes


# ------------------------------------------------------
# Function to draw bounding boxes on a video
# ------------------------------------------------------
def save_video_with_boxes(video_id):
    # Correct path for your structure
    input_path = os.path.join(VIDEO_DIR, video_id, "drone_video.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")

    if not os.path.exists(input_path):
        print(f"[ERROR] Video does not exist: {input_path}")
        return

    print(f"Processing {video_id}...")

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    boxes_for_video = ann_by_video.get(video_id, {})

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in boxes_for_video:
            for b in boxes_for_video[frame_idx]:
                cv2.rectangle(
                    frame,
                    (b["x1"], b["y1"]),
                    (b["x2"], b["y2"]),
                    (0, 255, 0),
                    2
                )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[DONE] Saved video: {output_path}")


# ------------------------------------------------------
# Run for ALL videos in the annotation JSON
# ------------------------------------------------------
for vid in video_ids:
    save_video_with_boxes(vid)

print("\nAll videos processed.")
