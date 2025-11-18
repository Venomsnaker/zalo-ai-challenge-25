"""Convert Zalo AI Challenge 2025 drone dataset to COCO format."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from PIL import Image
from tqdm import tqdm

CATEGORIES = [
    "Backpack",
    "Jacket",
    "Laptop",
    "Lifering",
    "MobilePhone",
    "Person1",
    "WaterBottle",
]


def load_annotations(annotations_path: Path) -> List[Dict]:
    """Load Zalo format annotations from JSON file."""
    with open(annotations_path) as f:
        return json.load(f)


def split_dataset(
    annotations: List[Dict], val_split: float = 0.15, test_split: float = 0.15
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split annotations into train/val/test sets deterministically."""
    total = len(annotations)
    test_size = int(total * test_split)
    val_size = int(total * val_split)

    sorted_anns = sorted(annotations, key=lambda x: hash(x["video_id"]))
    return (
        sorted_anns[test_size + val_size :],
        sorted_anns[test_size : test_size + val_size],
        sorted_anns[:test_size],
    )


def extract_frame(
    video_path: Path, frame_num: int, output_path: Path
) -> Tuple[int, int]:
    """Extract and save a single frame from video. Returns (width, height)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

        h, w = frame.shape[:2]
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(output_path)
        return w, h
    finally:
        cap.release()


def convert_bbox(
    x1: int, y1: int, x2: int, y2: int
) -> Optional[Tuple[float, float, float, float]]:
    """Convert bbox from (x1,y1,x2,y2) to COCO (x,y,w,h). Returns None if invalid."""
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    w, h = x2 - x1, y2 - y1
    return (float(x1), float(y1), float(w), float(h)) if w > 0 and h > 0 else None


def create_coco_categories() -> List[Dict]:
    """Generate COCO category definitions."""
    return [
        {"id": i, "name": cat, "supercategory": "object"}
        for i, cat in enumerate(CATEGORIES)
    ]


def process_split(
    annotations: List[Dict],
    samples_dir: Path,
    output_dir: Path,
    split: str,
    category_map: Dict[str, int],
) -> Dict:
    """Process a single split and generate COCO format data."""
    images, coco_anns = [], []
    img_id = ann_id = 0
    images_dir = output_dir / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_cache = {}

    for video_ann in tqdm(annotations, desc=f"{split:>5}", unit="vid"):
        video_id = video_ann["video_id"]
        video_path = samples_dir / video_id / "drone_video.mp4"

        if not video_path.exists():
            continue

        category_id = category_map.get(video_id.rsplit("_", 1)[0])
        if category_id is None:
            continue

        for ann_group in video_ann["annotations"]:
            for bbox_data in ann_group.get("bboxes", []):
                frame_num = bbox_data["frame"]
                img_name = f"{video_id}_frame_{frame_num:06d}.jpg"
                img_path = images_dir / img_name

                if img_name not in image_cache:
                    if not img_path.exists():
                        try:
                            w, h = extract_frame(video_path, frame_num, img_path)
                        except Exception:
                            continue
                    else:
                        w, h = Image.open(img_path).size

                    images.append(
                        {
                            "id": img_id,
                            "file_name": img_name,
                            "width": w,
                            "height": h,
                            "video_id": video_id,
                            "frame_number": frame_num,
                        }
                    )
                    image_cache[img_name] = img_id
                    img_id += 1

                bbox = convert_bbox(
                    bbox_data["x1"], bbox_data["y1"], bbox_data["x2"], bbox_data["y2"]
                )
                if bbox is None:
                    continue

                coco_anns.append(
                    {
                        "id": ann_id,
                        "image_id": image_cache[img_name],
                        "category_id": category_id,
                        "bbox": list(bbox),
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1

    return {
        "images": images,
        "annotations": coco_anns,
        "categories": create_coco_categories(),
    }


def convert_to_coco(
    train_dir: str = "train",
    output_dir: str = "coco_dataset",
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> None:
    """Convert Zalo AI Challenge dataset to COCO format."""
    train_path = Path(train_dir)
    output_path = Path(output_dir)
    samples_dir = train_path / "samples"
    annotations_file = train_path / "annotations" / "annotations.json"

    annotations = load_annotations(annotations_file)
    train_anns, val_anns, test_anns = split_dataset(annotations, val_split, test_split)

    category_map = {cat: i for i, cat in enumerate(CATEGORIES)}

    print(f"Converting {len(annotations)} videos to COCO format")
    print(f"Split: train={len(train_anns)}, val={len(val_anns)}, test={len(test_anns)}")

    for split_name, split_data in [
        ("train", train_anns),
        ("val", val_anns),
        ("test", test_anns),
    ]:
        coco_data = process_split(
            split_data, samples_dir, output_path, split_name, category_map
        )

        ann_dir = output_path / split_name / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        with open(ann_dir / "instances.json", "w") as f:
            json.dump(coco_data, f, indent=2)

        print(
            f"{split_name:>5}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations"
        )

    print(f"\nDataset saved to {output_path}")


def main():
    """Entry point."""
    convert_to_coco(
        train_dir="train", output_dir="coco_dataset", val_split=0.15, test_split=0.15
    )


if __name__ == "__main__":
    main()
