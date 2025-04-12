import matplotlib.pyplot as plt
from bounding_box import results

# Frame-wise IoU data
frame_ids = [row[0] for row in results]
iou_scores = [row[1] for row in results]
average_iou = sum(iou_scores) / len(iou_scores)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(frame_ids, iou_scores, marker='o', color='orange', label='IoU per Frame')
plt.axhline(y=average_iou, color='blue', linestyle='--', label=f'Average IoU = {average_iou:.2f}')
plt.title("IoU per Frame with Average IoU Line")
plt.xlabel("Frame ID")
plt.ylabel("IoU Score")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show plot
plt.show()

# Optional: Save the plot
plt.savefig("results/iou_plot.png")
print("IoU plot saved as 'results/iou_plot.png'")

