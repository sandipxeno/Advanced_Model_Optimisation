from onnxruntime.quantization import quantize_dynamic, QuantType

# Apply dynamic quantization
quantized_model_path = "C:/Users/user/Desktop/yolov12_quant_api/models/yolo12n.quantized.onnx"
quantize_dynamic("C:/Users/user/Desktop/yolov12_quant_api/models/distilled_pruned_model/weights/best.onnx", quantized_model_path, weight_type=QuantType.QUInt8)

print(f"Quantized model saved at {quantized_model_path}")
