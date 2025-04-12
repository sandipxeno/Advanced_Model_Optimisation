import os
import cv2
import csv
from iou_utils import calculate_iou
from visualize_iou import draw_boxes

# Simulated bounding box extractor for a given frame
# Replace with YOLO model inference + ground truth fetch

def get_predictions_for_frame(frame_id):
    return [50, 50, 200, 200] if frame_id % 2 == 0 else [30, 30, 180, 180]

def get_ground_truth_for_frame(frame_id):
    return [60, 60, 190, 190] if frame_id % 2 == 0 else [35, 35, 175, 175]

video_path = "/content/videoplayback_output.mp4"  # Adjust path based on where the video is stored
cap = cv2.VideoCapture(video_path)

frame_id = 0
ious = []
results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pred_box = get_predictions_for_frame(frame_id)
    gt_box = get_ground_truth_for_frame(frame_id)
    iou = calculate_iou(pred_box, gt_box)
    draw_boxes(frame, gt_box, pred_box, iou)

    results.append([frame_id, iou])
    ious.append(iou)
    frame_id += 1

cap.release()

# Save to CSV
os.makedirs("results", exist_ok=True)
with open("results/iou_video_scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["FrameID", "IoU"])
    writer.writerows(results)

print(f"Average IoU across video: {sum(ious)/len(ious):.2f}")
