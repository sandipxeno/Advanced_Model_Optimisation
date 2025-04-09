from onnxruntime.quantization import quantize_dynamic, QuantType

# Apply dynamic quantization
quantized_model_path = "D:/prodigal-4/models/yolo12n.quantized.onnx"
quantize_dynamic("D:/prodigal-4/models/yolo12n.onnx", quantized_model_path, weight_type=QuantType.QUInt8)

print(f"Quantized model saved at {quantized_model_path}")
