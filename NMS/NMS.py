import numpy as np

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def non_max_suppression(boxes, scores, iou_threshold):
    indices = np.argsort(scores)[::-1]  # Sort scores in descending order
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        rest = indices[1:]

        filtered = []
        for i in rest:
            if iou(boxes[current], boxes[i]) < iou_threshold:
                filtered.append(i)

        indices = np.array(filtered)

    return keep

boxes = [
    [100, 100, 210, 210],
    [105, 105, 215, 215],
    [250, 250, 400, 400]
]
scores = [0.9, 0.8, 0.7]

iou_threshold = 0.5
selected_indices = non_max_suppression(boxes, scores, iou_threshold)
selected_boxes = [boxes[i] for i in selected_indices]

print("Selected Boxes:", selected_boxes)