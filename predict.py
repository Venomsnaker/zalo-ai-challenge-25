import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from config import *  # Import tất cả cấu hình
import torchvision.transforms as T


class YOLODetector:
    def __init__(self, model_path, confidence=CONFIDENCE_THRESHOLD, use_dinov2=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.conf = confidence
        self.model.to(self.device)

        self.frame_cache = []
        self.max_cache_size = 5

        # DINOv2 cho postprocessing
        self.use_dinov2 = use_dinov2
        if use_dinov2:
            self._init_dinov2()
            self.ref_embeddings = None  # Sẽ được set từ bên ngoài
            self.sim_threshold = 0.01

    def _init_dinov2(self):
        """Khởi tạo DINOv2 model"""
        try:
            from transformers import AutoImageProcessor, AutoModel

            print("Using HuggingFace DINOv2-small for postprocessing")
            self.dino_processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-small"
            )
            self.dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(
                self.device
            )
            self.hf_available = True
        except Exception:
            print("Using torch.hub dinov2_vits14 for postprocessing")
            self.dino_processor = T.Compose(
                [
                    T.Resize((518, 518)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
                ]
            )
            self.dino_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
            )
            self.dino_model.eval().to(self.device)
            self.hf_available = False

    def set_reference_embeddings(self, video_id, object_images_folder):
        """Load reference embeddings từ object_images của video"""
        if not self.use_dinov2:
            return

        self.ref_embeddings = []
        for i in range(1, 4):
            img_path = os.path.join(object_images_folder, f"img_{i}.jpg")
            if os.path.exists(img_path):
                pil_img = Image.open(img_path).convert("RGB")
                emb = self._image_to_embedding(pil_img)
                self.ref_embeddings.append(emb)

        print(f"Loaded {len(self.ref_embeddings)} reference embeddings for {video_id}")

    def _image_to_embedding(self, pil_image):
        """Convert PIL image to DINOv2 embedding"""
        if self.hf_available:
            inputs = self.dino_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.dino_model(**inputs)
                emb = out.last_hidden_state[:, 0, :]
                emb = torch.nn.functional.normalize(emb, dim=-1)
            return emb.cpu().numpy().reshape(-1)
        else:
            tensor = self.dino_processor(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.dino_model.forward_features(tensor)
                if isinstance(out, dict):
                    emb = out.get(
                        "x_norm_clstoken",
                        next(v for v in out.values() if isinstance(v, torch.Tensor)),
                    )
                else:
                    emb = out[:, 0, :] if out.ndim == 3 else out
                emb = torch.nn.functional.normalize(emb, dim=-1)
            return emb.cpu().numpy().reshape(-1)

    def _cosine_similarity(self, a, b):
        """Tính cosine similarity"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _dinov2_filter(self, frame_rgb_np, bbox):
        """
        Filter bbox bằng DINOv2 similarity với reference images
        Returns: True nếu bbox pass filter, False nếu không
        """
        if (
            not self.use_dinov2
            or self.ref_embeddings is None
            or len(self.ref_embeddings) == 0
        ):
            return True  # Không filter nếu không có ref embeddings

        x1, y1, x2, y2 = bbox
        crop = frame_rgb_np[y1:y2, x1:x2]

        if crop.size == 0:
            return False

        # Convert crop to PIL Image
        pil_crop = Image.fromarray(crop)
        crop_emb = self._image_to_embedding(pil_crop)

        # Tính similarity với tất cả reference embeddings
        best_sim = max(
            self._cosine_similarity(crop_emb, ref) for ref in self.ref_embeddings
        )

        return best_sim >= self.sim_threshold

    def predict_streaming(self, frame_rgb_np, frame_idx):
        """
        Hàm predict cho streaming mode với DINOv2 postprocessing
        """
        # Cache frame hiện tại
        self.frame_cache.append({"frame_idx": frame_idx, "frame": frame_rgb_np.copy()})

        if len(self.frame_cache) > self.max_cache_size:
            self.frame_cache.pop(0)

        # Chạy YOLO detection
        results = self.model.predict(
            source=frame_rgb_np, conf=self.conf, iou=0.5, imgsz=640, verbose=False
        )

        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                # Lấy tất cả boxes và confidences
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()

                # Sort theo confidence giảm dần
                sorted_indices = np.argsort(confidences)[::-1]

                # Thử từng box theo thứ tự confidence, trả về box đầu tiên pass DINOv2 filter
                for idx in sorted_indices:
                    box = boxes[idx]
                    x1, y1, x2, y2 = map(int, box)
                    bbox = [x1, y1, x2, y2]

                    # === POSTPROCESSING: DINOv2 filtering ===
                    if self._dinov2_filter(frame_rgb_np, bbox):
                        # Cache bbox để dùng cho temporal smoothing (nếu cần)
                        self.frame_cache[-1]["bbox"] = bbox
                        return bbox

        return None

    def detect_video(self, video_path):
        """
        Chạy model trên video và trả về tất cả các detection tìm thấy (raw boxes).
        """
        if not os.path.exists(video_path):
            print(f"Warning: File {video_path} not found.")
            return []

        detections = []
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox = self.predict_streaming(frame_rgb, frame_idx)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                detections.append(
                    {
                        "frame": frame_idx,
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    }
                )
            frame_idx += 1

        cap.release()
        return detections

    def process_test_set(self, test_root, output_json):
        """
        Xử lý test set với DINOv2 postprocessing
        """
        video_dirs = sorted([d for d in Path(test_root).iterdir() if d.is_dir()])
        submission = []

        print(f"Starting processing on {self.device} with confidence={self.conf}...")

        for i, vid_dir in enumerate(video_dirs, 1):
            video_path = vid_dir / "drone_video.mp4"
            print(f"\n[{i}/{len(video_dirs)}] 🎥 {vid_dir.name}", end="")

            # Reset cache
            self.frame_cache = []

            # Load reference embeddings cho video này
            object_images_folder = vid_dir / "object_images"
            if object_images_folder.exists():
                self.set_reference_embeddings(vid_dir.name, object_images_folder)
            else:
                print(f" (No reference images found)")
                self.ref_embeddings = None

            # Xử lý video
            cap = cv2.VideoCapture(str(video_path))
            detections = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bbox = self.predict_streaming(frame_rgb, frame_idx)

                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    detections.append(
                        {"frame": frame_idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    )

                frame_idx += 1

            cap.release()

            submission.append(
                {
                    "video_id": vid_dir.name,
                    "detections": [{"bboxes": detections}] if detections else [],
                }
            )
            print(f" -> Found {len(detections)} boxes.")

        os.makedirs(os.path.dirname(output_json), exist_ok=True)

        with open(output_json, "w") as f:
            json.dump(submission, f, indent=2)

        print(f"\nProcess completed. Saved to: {output_json}")
        return submission


def run_prediction():
    print("--- Bắt đầu Dự đoán (Inference) ---")

    # Kiểm tra và lấy mô hình đã train (hoặc dùng mô hình Kaggle input)
    # Lấy đường dẫn của mô hình tốt nhất đã train
    best_model_path = os.path.join(PROJECT_DIR, "finetune", "weights", "best.pt")

    if os.path.exists(best_model_path):
        final_model_path = best_model_path
        print(f"Sử dụng mô hình đã train: {final_model_path}")
    elif os.path.exists(TRAINED_MODEL_PATH):
        final_model_path = TRAINED_MODEL_PATH
        print(f"Sử dụng mô hình có sẵn: {final_model_path}")
    else:
        print("Lỗi: Không tìm thấy mô hình nào để dự đoán.")
        return

    detector = YOLODetector(final_model_path, confidence=CONFIDENCE_THRESHOLD)

    submission = detector.process_test_set(test_root=TEST_ROOT, output_json=OUTPUT_PATH)


if __name__ == "__main__":
    run_prediction()
