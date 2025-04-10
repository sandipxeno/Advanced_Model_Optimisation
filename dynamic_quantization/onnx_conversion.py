from ultralytics import YOLO

model = YOLO("C:/Users/user/Desktop/yolov12_quant_api/models/distilled_pruned_model/weights/best.pt")

model.export(format="onnx", dynamic=True, opset=12) 