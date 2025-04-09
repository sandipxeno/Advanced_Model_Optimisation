import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('D:/prodigal-4/yolov12_quant_api/models/yolo12n.pt')  # Your pretrained model

# Access the PyTorch model inside
model_torch = model.model

# Apply global unstructured pruning
parameters_to_prune = []
for module in model_torch.modules():
    if isinstance(module, nn.Conv2d):
        parameters_to_prune.append((module, 'weight'))

# Apply L1 unstructured pruning globally with 30% sparsity
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,  # 30% of weights pruned globally
)

# Optional: Remove pruning re-param so weights are fixed
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

# Save the pruned model
torch.save(model_torch.state_dict(), 'yolo12n_pruned.pth')
print("Pruned model saved as 'yolo12n_pruned.pth'")
