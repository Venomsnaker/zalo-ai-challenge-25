import os
import json
import random
import cv2
import yaml
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import * # Import tất cả cấu hình

def prepare_dirs(base_dir):
    """Tạo thư mục images và labels cho tập train/val."""
    imgs = base_dir / "images"
    lbls = base_dir / "labels"
    os.makedirs(imgs, exist_ok=True)
    os.makedirs(lbls, exist_ok=True)
    return str(imgs), str(lbls)

def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height, class_id):
    """Chuyển đổi tọa độ [x1, y1, x2, y2] sang định dạng YOLO [class_id x_center y_center width height] (chuẩn hóa 0-1)."""
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Chuẩn hóa và giới hạn trong khoảng [0, 1]
    x_center = max(0, min(1, x_center * dw))
    y_center = max(0, min(1, y_center * dh))
    width = max(0, min(1, width * dw))
    height = max(0, min(1, height * dh))
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def extract_frames_and_labels(video_id, mode="train", ann_dict=None):
    """
    Trích xuất các frame có bounding box và các frame nền (background) định kỳ, 
    lưu thành cặp ảnh (.jpg) và nhãn (.txt) theo định dạng YOLO.
    """
    if ann_dict is None:
        raise ValueError("ann_dict must be provided.")

    video_dir = os.path.join(SAMPLES_DIR, video_id)
    video_path = os.path.join(video_dir, "drone_video.mp4")
    
    if not os.path.exists(video_path):
        return f"Missing video: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Cannot open: {video_path}"

    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_annotations = {}
    video_record = ann_dict.get(video_id, {})
    
    # Tổng hợp tất cả bboxes theo frame_num
    for interval in video_record.get("annotations", []):
        for bbox_item in interval.get("bboxes", []):
            frame_num = bbox_item['frame']
            if frame_num not in frame_annotations:
                frame_annotations[frame_num] = []
            frame_annotations[frame_num].append(bbox_item)

    img_out = TRAIN_IMG_DIR if mode == "train" else VAL_IMG_DIR
    lbl_out = TRAIN_LBL_DIR if mode == "train" else VAL_LBL_DIR
    
    saved_obj = 0
    saved_bg = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        should_save = False
        is_background = False

        if frame_idx in frame_annotations:
            should_save = True
            is_background = False
        
        elif frame_idx % EMPTY_FRAME_STEP == 0:
            should_save = True
            is_background = True
        
        if should_save:
            image_name_stem = f"{video_id}_frame_{frame_idx:06d}"
            img_path = os.path.join(img_out, image_name_stem + ".jpg")
            txt_path = os.path.join(lbl_out, image_name_stem + ".txt")

            # Lưu ảnh
            success = cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                print(f"Failed to write: {img_path}")
                frame_idx += 1
                continue

            # Lưu Label
            if not is_background:
                yolo_labels = []
                for bbox in frame_annotations[frame_idx]:
                    yolo_line = convert_to_yolo_format(
                        bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'], 
                        img_width, img_height, 0 # Class ID là 0 vì chỉ có 1 class 'target'
                    )
                    yolo_labels.append(yolo_line)
                
                with open(txt_path, "w") as f:
                    f.write('\n'.join(yolo_labels))
                saved_obj += 1
            else:
                # Ghi file rỗng cho ảnh nền (Background image)
                with open(txt_path, "w") as f:
                    pass 
                saved_bg += 1
        
        frame_idx += 1

    cap.release()
    return f"{video_id}: {saved_obj} annotated + {saved_bg} background frames saved."

def run_preprocessing():
    print("--- Bắt đầu Tiền xử lý Dữ liệu ---")
    
    with open(ANNOTATIONS_PATH, "r") as f:
        annotations = json.load(f)

    video_ids = [a["video_id"] for a in annotations]
    ann_dict = {a["video_id"]: a for a in annotations}

    random.seed(42)
    random.shuffle(video_ids)
    split_idx = int(len(video_ids) * TRAIN_RATIO)
    train_videos = video_ids[:split_idx]
    val_videos = video_ids[split_idx:]
    print(f"Loaded {len(video_ids)} videos — Train: {len(train_videos)}, Val: {len(val_videos)}")

    # 2. Chuẩn bị thư mục output
    prepare_dirs(Path(WORK_DIR) / "train")
    prepare_dirs(Path(WORK_DIR) / "val")
    print(f"Đã tạo cấu trúc thư mục tại: {WORK_DIR}")

    print("\nExtracting frames + labels ...")
    futures = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for vid in train_videos:
            futures.append(ex.submit(extract_frames_and_labels, vid, "train", ann_dict))
        for vid in val_videos:
            futures.append(ex.submit(extract_frames_and_labels, vid, "val", ann_dict))
        
        for i, fut in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(futures)}] {fut.result()}")

    print("All videos processed!")

    train_imgs = glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg"))
    val_imgs = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg"))
    print(f"\n  Tổng Train images: {len(train_imgs)}")
    print(f"  Tổng Val images: {len(val_imgs)}")
    
    data_yaml = {
        "train": str(TRAIN_IMG_DIR.absolute()),
        "val": str(VAL_IMG_DIR.absolute()),
        "nc": len(CLASS_NAMES),                 
        "names": CLASS_NAMES
    }
    with open(DATA_YAML_PATH, "w") as f:
        yaml.dump(data_yaml, f)
    print(f"\ndata.yaml created at: {DATA_YAML_PATH}")

if __name__ == "__main__":
    run_preprocessing()