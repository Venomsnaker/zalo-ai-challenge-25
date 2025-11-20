import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "mps"

SIM_THRESHOLD = 0.01   # keep all similarities >= 1%
ROOT = "data/train/samples"
JSON_INPUT = "data/output/submission_train.json"
JSON_OUTPUT = "data/output/submission_train_filtered.json"

HF_AVAILABLE = False

# ------------------------------------------------------------
# Try HuggingFace DINOv2-small first
# ------------------------------------------------------------
try:
    from transformers import AutoImageProcessor, AutoModel
    print("Using HuggingFace DINOv2-small")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE)
    HF_AVAILABLE = True
except Exception:
    print("HuggingFace not available → using torch.hub dinov2_vits14 (small)")
    processor = None
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model.eval().to(DEVICE)


# ------------------------------------------------------------
# Image → CLS Embedding
# ------------------------------------------------------------
def image_to_embedding(preproc, model, pil_image):
    """Returns L2-normalized CLS embedding (numpy 1D)."""

    if HF_AVAILABLE:
        # HuggingFace version
        inputs = preproc(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :]       # CLS token
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy().reshape(-1)

    else:
        # torch.hub version (DINOv2)
        tensor = preproc(pil_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model.forward_features(tensor)

            if isinstance(out, dict):
                # Most modern DINOv2 hub versions
                if "x_norm_clstoken" in out:
                    emb = out["x_norm_clstoken"]
                else:
                    # fallback: first tensor field
                    emb = next(v for v in out.values() if isinstance(v, torch.Tensor))
            else:
                # older versions
                if out.ndim == 3:
                    emb = out[:, 0, :]
                else:
                    emb = out

            emb = torch.nn.functional.normalize(emb, dim=-1)
            return emb.cpu().numpy().reshape(-1)


# ------------------------------------------------------------
# Similarity
# ------------------------------------------------------------
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ------------------------------------------------------------
# Load reference embeddings
# ------------------------------------------------------------
def load_reference_embeddings(video_id):
    folder = f"{ROOT}/{video_id}/object_images"

    if HF_AVAILABLE:
        preproc = processor
    else:
        # DINOv2 small expects 518×518, same as base
        preproc = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    refs = []

    for i in range(1, 4):
        path = os.path.join(folder, f"img_{i}.jpg")
        if not os.path.exists(path):
            continue

        pil = Image.open(path).convert("RGB")
        emb = image_to_embedding(preproc, model, pil)
        refs.append(emb)

    return refs, preproc


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    with open(JSON_INPUT, "r") as f:
        data = json.load(f)

    output_data = []

    for entry in data:
        video_id = entry["video_id"]
        print(f"\n▶ Processing {video_id}")

        video_path = f"{ROOT}/{video_id}/drone_video.mp4"
        ref_embs, preproc = load_reference_embeddings(video_id)

        if len(ref_embs) == 0:
            print("No reference images. Skipping.")
            continue

        cap = cv2.VideoCapture(video_path)

        # map frame → list of boxes
        frame_map = {}
        for det in entry["detections"]:
            for b in det["bboxes"]:
                frame_map.setdefault(b["frame"], []).append(b)

        filtered_bboxes = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx not in frame_map:
                frame_idx += 1
                continue

            for box in frame_map[frame_idx]:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                pil_crop = Image.fromarray(crop[:, :, ::-1])  # BGR→RGB
                emb = image_to_embedding(preproc, model, pil_crop)

                # check similarity against all refs
                best_sim = max(cosine(emb, ref) for ref in ref_embs)

                if best_sim >= SIM_THRESHOLD:
                    filtered_bboxes.append(box)

                print(f"Frame {frame_idx} sim={best_sim:.3f}")

            frame_idx += 1

        cap.release()

        output_data.append({
            "video_id": video_id,
            "detections": [
                {"bboxes": filtered_bboxes}
            ]
        })

    with open(JSON_OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2)

    print("\n✅ Done. Saved:", JSON_OUTPUT)


if __name__ == "__main__":
    main()
