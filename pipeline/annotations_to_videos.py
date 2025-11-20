import json
import cv2
import os

# ------------------------------------------------------
# Config
# ------------------------------------------------------
ANNOT_FILE = "data/train/annotations/annotations.json"
VIDEO_DIR = "data/train/samples"
OUTPUT_DIR = "data/output/videos/groundtruth"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------
# Load annotations (preserve detection segments)
# ------------------------------------------------------
with open(ANNOT_FILE, "r") as f:
    annotations = json.load(f)

ann_by_video = {}
video_ids = []

for entry in annotations:
    vid = entry["video_id"]
    video_ids.append(vid)

    tracks = []  # each detection group is one track segment

    for det in entry["annotations"]:
        track = {}  # frame -> list of boxes for this detection segment

        for box in det["bboxes"]:
            frame = box["frame"]
            if frame not in track:
                track[frame] = []
            track[frame].append(box)

        tracks.append(track)

    ann_by_video[vid] = tracks


# ------------------------------------------------------
# Drawing settings
# ------------------------------------------------------
BOX_COLOR = (0, 255, 0)  # same color for all boxes (green)
THICKNESS = 2


# ------------------------------------------------------
# Function to draw bounding boxes on a video
# ------------------------------------------------------
def save_video_with_boxes(video_id):

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
    tracks = ann_by_video.get(video_id, [])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # For each track segment (detection group)
        for track in tracks:
            if frame_idx in track:
                for b in track[frame_idx]:
                    cv2.rectangle(
                        frame,
                        (b["x1"], b["y1"]),
                        (b["x2"], b["y2"]),
                        BOX_COLOR,
                        THICKNESS
                    )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[DONE] Saved video: {output_path}")


# ------------------------------------------------------
# Process ALL videos in the JSON
# ------------------------------------------------------
for vid in video_ids:
    save_video_with_boxes(vid)

print("\nAll videos processed.")
