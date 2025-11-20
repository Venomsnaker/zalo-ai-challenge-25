import csv
from pipeline.utils import compute_st_iou

def main():
    ground_truth_file = "data/train/annotations/annotations.json"
    submission_file = "data/output/submission_train.json"
    submission_filtered_file = "data/output/submission_train_filtered.json"
    output_csv_file = "data/output/st_iou_comparison.csv"
    
    results = compute_st_iou(ground_truth_file, submission_file)
    results_filtered = compute_st_iou(ground_truth_file, submission_filtered_file)

    # Prepare data for CSV
    rows = []
    total_orig = 0
    total_filt = 0
    count_orig = 0
    count_filt = 0
    total_improvement = 0
    count_improvement = 0

    for vid in sorted(results.keys()):
        st_iou_orig = results[vid]
        st_iou_filt = results_filtered.get(vid, None)
        
        if st_iou_orig is not None:
            total_orig += st_iou_orig
            count_orig += 1
        
        if st_iou_filt is not None:
            total_filt += st_iou_filt
            count_filt += 1

        if st_iou_filt is None:
            improvement = None
        else:
            improvement = st_iou_filt - st_iou_orig
            total_improvement += improvement
            count_improvement += 1

        rows.append({
            "video_id": vid,
            "original_iou": st_iou_orig,
            "filtered_iou": st_iou_filt,
            "improvement": improvement
        })

    # Write to CSV
    with open(output_csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "original_iou", "filtered_iou", "improvement"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Compute and print averages
    avg_orig = total_orig / count_orig if count_orig > 0 else 0
    avg_filt = total_filt / count_filt if count_filt > 0 else 0
    avg_improvement = total_improvement / count_improvement if count_improvement > 0 else 0

    print(f"Average original ST-IoU: {avg_orig:.4f}")
    print(f"Average filtered ST-IoU: {avg_filt:.4f}")
    print(f"Average improvement: {avg_improvement:.4f}")
    print(f"ST-IoU comparison saved to {output_csv_file}")

if __name__ == "__main__":
    main()
