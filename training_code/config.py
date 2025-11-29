import os
from pathlib import Path

# Docker paths for Zalo AI Challenge
DATASET_ROOT = "/data/train"
TEST_ROOT = "/data/samples"
# Model path inside docker
TRAINED_MODEL_PATH = "/saved_models/yoloe-11l-50epochs.pt"
ANNOTATIONS_PATH = os.path.join(DATASET_ROOT, "annotations/annotations.json")
SAMPLES_DIR = os.path.join(DATASET_ROOT, "samples")

# Temporary working directories for Docker environment
WORK_DIR = "/tmp/stream_dataset"
os.makedirs(WORK_DIR, exist_ok=True)

PROJECT_DIR = "/tmp/yolo_finetune"
os.makedirs(PROJECT_DIR, exist_ok=True)

# Output path
os.makedirs("/result", exist_ok=True)
OUTPUT_PATH = "/result/submission.json"

TRAIN_RATIO = 0.85
EMPTY_FRAME_STEP = 30
CLASS_NAMES = ["target"]

TRAIN_IMG_DIR = Path(WORK_DIR) / "train" / "images"
TRAIN_LBL_DIR = Path(WORK_DIR) / "train" / "labels"
VAL_IMG_DIR = Path(WORK_DIR) / "val" / "images"
VAL_LBL_DIR = Path(WORK_DIR) / "val" / "labels"
DATA_YAML_PATH = Path(WORK_DIR) / "data.yaml"

EPOCHS = 50
BATCH = 16
NUM_WORKERS = 4
MODEL_YAML = "yoloe-11l.yaml"
MODEL_WEIGHTS = "yoloe-11l-seg.pt"

CONFIDENCE_THRESHOLD = 0.05
