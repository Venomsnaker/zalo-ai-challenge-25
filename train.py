import os
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer
from config import *

def run_training():
    print("--- Bắt đầu Huấn luyện Mô hình ---")

    print(f"Đang load model từ config: {MODEL_YAML}, weights: {MODEL_WEIGHTS}")
    model = YOLOE(MODEL_YAML)
    model.load(MODEL_WEIGHTS)
    model.set_classes(CLASS_NAMES)

    model_info = model.info()
    print(f"Model Summary: {model_info[0]} layers, {model_info[1]} parameters, {model_info[3]} GFLOPs")

    print("\nStarting YOLO training...")
    
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Lỗi: Không tìm thấy data.yaml tại {DATA_YAML_PATH}. Hãy chạy preprocessing.py trước.")
        return

    results = model.train(
        data=str(DATA_YAML_PATH),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=640,
        workers=NUM_WORKERS,          
        patience=10,        
        exist_ok=True,      
        project=PROJECT_DIR,
        trainer=YOLOEPETrainer,
        name="finetune", 
    )
    print("\nTraining completed")
    
    best_model_path = os.path.join(PROJECT_DIR, "finetune", "weights", "best.pt")
    if os.path.exists(best_model_path):
        print(f"Mô hình tốt nhất được lưu tại: {best_model_path}")
    else:
        print("Không tìm thấy best.pt. Kiểm tra log huấn luyện.")

if __name__ == "__main__":
    run_training()