from visual_based_pipeline import VisualBasedPipeline
from utils import load_data

def main():
    data = load_data("data/train/samples")
    visual_pipeline = VisualBasedPipeline(
        reference_model_path="data/models/yoloe-11s-seg.pt",
        footage_model_path="data/models/yoloe-11s-seg.pt",
    )
    visual_pipeline.run(data, output_path="data/output/yoloe_vb_1.json")
    

if __name__ == "__main__":
    main()
