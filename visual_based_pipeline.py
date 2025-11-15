from pathlib import Path    
import json

import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

class VisualBasedPipeline:
    def __init__(
        self,
        reference_model_path="data/models/yoloe-11s-seg.pt",
        footage_model_path="data/models/yoloe-11s-seg.pt",
    ):
        try:
            self.reference_model = YOLOE(reference_model_path).to("cuda")
            self.footage_model = YOLOE(footage_model_path).to("cuda")
            print("YOLOE models successfully moved to CUDA.")
        except Exception as e:
            print(f"Failed to move models to CUDA: {e}. Falling back to CPU.")
            self.reference_model = YOLOE(reference_model_path)
            self.footage_model = YOLOE(footage_model_path)
        
    def _extract_bbox(self, res) -> list[int]:
        bbox = 0
        detections = res[0].boxes
        
        if len(detections) > 0:
            try:
                bbox = detections.xyxy[0].cpu().numpy().astype(int).tolist()
            except:
                bbox = detections.xyxy[0][0].astype(int).tolist() if len(detections.xyxy[0]) > 0 else None
        else:
            img_height, img_width = res[0].orig_shape
            bbox = [0, 0, img_width, img_height]
        return bbox
    
    def _process_footage_result(self, res_footage, frame_offset=0) -> list[dict]:
        bboxes = []
        
        for frame_idx, frame_res in enumerate(res_footage):
            detections = frame_res.boxes
            
            if len(detections) > 0:
                try:
                    bbox = detections.xyxy[0].cpu().numpy().astype(int).tolist()
                except:
                    bbox = detections.xyxy[0][0].astype(int).tolist() if len(detections.xyxy[0]) > 0 else None
                
                # Format bbox to JSON format
                x1, y1, x2, y2 = bbox
                bbox_json = {
                        "frame": frame_idx + frame_offset,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                bboxes.append(bbox_json)
        return bboxes
                
        
    def predict_ref(self, ref_path):
        ref_path = ref_path
        res = self.reference_model.predict(ref_path)
        return res
    
    def predict_footage(self, footage_path, ref_path, ref_bbox):
        visual_prompts = dict(
            bboxes=np.array([ref_bbox]),
            cls=np.array([0])
        )
        res = self.footage_model.predict(
            footage_path,
            refer_image=ref_path,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
        )
        return res
        
    def run(self, data, output_path=""):
        data = data[:1] # Remove me
        res = []
        
        for item in data:
            video_id = item['id']
            footage_path = item['footage_path']
            classes = [item['object_name']]
            imgs = item['images']
            
            # Set classes for reference labeling model
            self.reference_model.set_classes(classes, self.reference_model.get_text_pe(classes))
            
            # Use the first image as reference
            ref_img_path = list(imgs.values())[0]
            res_ref = self.predict_ref(ref_img_path)
            ref_bbox = self._extract_bbox(res_ref)
            
            # Annotate footage
            res_footage = self.predict_footage(
                footage_path=footage_path,
                ref_path=ref_img_path,
                ref_bbox=ref_bbox,
            )
            footage_annotations = self._process_footage_result(res_footage)
            res.append(
                {
                    "video_id": video_id,
                    "detections": [{"bboxes": footage_annotations}],
                }
            )
        
        # Save to JSON format
        if output_path != "":
            with open(output_path, 'w') as f:
                json.dump(res, f)
        return res
        
