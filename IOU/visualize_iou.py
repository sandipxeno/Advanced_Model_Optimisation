import cv2
import matplotlib.pyplot as plt

def draw_boxes(image, gt_box, pred_box, iou):
    """
    Draws ground truth (blue) and predicted (green) boxes with IoU overlay.
    """
    img = image.copy()
    cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, 0), 2)  # Ground truth
    cv2.rectangle(img, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 255, 0), 2)  # Prediction
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"IoU: {iou:.2f}")
    plt.axis('off')
    plt.show()
