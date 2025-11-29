# Training Code & Description

## Idea Description
Our solution employs a robust two-stage pipeline combining a high-performance object detector with a semantic similarity-based post-processing filter to accurately identify targets in drone videos.

### 1. Object Detection with YOLO
We utilize a fine-tuned YOLO (You Only Look Once) model as our primary object detector. The model is trained on the provided dataset to detect potential targets within the video frames. This stage provides high-recall candidate bounding boxes.

### 2. Post-processing with DINOv2
To further refine the detection results and reduce false positives, we implement a post-processing step using **DINOv2 (specifically `dinov2-small`)**, a self-supervised vision transformer model by Meta.

The process works as follows:
- **Reference Embeddings**: For each video, we extract feature embeddings from the provided reference object images (`object_images`) using the DINOv2 model. These embeddings serve as the "fingerprint" of the target object.
- **Candidate Filtering**: For every bounding box predicted by the YOLO model, we crop the corresponding region from the frame and extract its DINOv2 embedding.
- **Similarity Matching**: We calculate the cosine similarity between the candidate's embedding and the reference embeddings.
- **Thresholding**: Detections with a similarity score below a specific threshold are discarded. This ensures that only objects semantically similar to the reference targets are retained, significantly improving precision.

This approach leverages the strong feature representation capabilities of DINOv2 to handle variations in scale, lighting, and viewpoint, ensuring robust tracking of the specific target object.

## Training Code & Data
URL to download training data and models: [Insert URL here]

## Reproducibility
Seed used: 42
