import os
import time
import torch
import cv2
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from ultralytics import YOLO

video_dir = r"C:\Users\user\Desktop\ADVANCED_MODEL_OPTIMISATION\test(out_put)"
results_file = r"C:\Users\user\Desktop\ADVANCED_MODEL_OPTIMISATION\benchmarking\results.txt"

models = {
    "Base Model": r"C:\Users\user\Desktop\ADVANCED_MODEL_OPTIMISATION\models\yolo12n.pt",
    "Distilled + Pruned": r"C:\Users\user\Desktop\ADVANCED_MODEL_OPTIMISATION\models\distilled_pruned_model\weights\best.pt",
    "Quantized ONNX": r"C:\Users\user\Desktop\ADVANCED_MODEL_OPTIMISATION\models\yolo12n.quantized.onnx"
}

metrics = defaultdict(dict)

def evaluate_pytorch_model(model_path, video_paths):
    model = YOLO(model_path)
    total_time, frame_count = 0, 0

    for video in tqdm(video_paths, desc=f"Evaluating PyTorch: {os.path.basename(model_path)}"):
        cap = cv2.VideoCapture(str(video))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            start = time.time()
            _ = model(frame, verbose=False)
            total_time += time.time() - start
            frame_count += 1
        cap.release()

    latency = total_time / frame_count
    fps = frame_count / total_time
    return round(latency, 4), round(fps, 2), frame_count, round(total_time, 2)

def evaluate_onnx_model(onnx_path, video_paths):
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    total_time, frame_count = 0, 0

    for video in tqdm(video_paths, desc=f"Evaluating ONNX: {os.path.basename(onnx_path)}"):
        cap = cv2.VideoCapture(str(video))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.resize(frame, (640, 640))
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = img / 255.0
            img = img.astype('float32')
            img = img.reshape(1, 3, 640, 640)

            start = time.time()
            _ = ort_session.run(None, {input_name: img})
            total_time += time.time() - start
            frame_count += 1
        cap.release()

    latency = total_time / frame_count
    fps = frame_count / total_time
    return round(latency, 4), round(fps, 2), frame_count, round(total_time, 2)

def get_model_size(path):
    return round(os.path.getsize(path) / (1024 * 1024), 2)  # MB

def main():
    video_paths = list(Path(video_dir).glob("*.mp4"))
    if not video_paths:
        print("No test videos found.")
        return

    for name, path in models.items():
        if path.endswith(".onnx"):
            latency, fps, frames, total_time = evaluate_onnx_model(path, video_paths)
        else:
            latency, fps, frames, total_time = evaluate_pytorch_model(path, video_paths)
        metrics[name]["Latency (s/frame)"] = latency
        metrics[name]["FPS"] = fps
        metrics[name]["Frames"] = frames
        metrics[name]["Total Time (s)"] = total_time
        metrics[name]["Model Size (MB)"] = get_model_size(path)

    # Format Output
    headers = ["Model", "Latency (s/frame)", "FPS", "Frames", "Total Time (s)", "Model Size (MB)"]
    col_widths = [25, 18, 10, 10, 17, 17]

    with open(results_file, 'w') as f:
        # Header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")

        # Rows
        for model_name, result in metrics.items():
            row = [
                model_name.ljust(col_widths[0]),
                str(result["Latency (s/frame)"]).ljust(col_widths[1]),
                str(result["FPS"]).ljust(col_widths[2]),
                str(result["Frames"]).ljust(col_widths[3]),
                str(result["Total Time (s)"]).ljust(col_widths[4]),
                str(result["Model Size (MB)"]).ljust(col_widths[5])
            ]
            f.write(" | ".join(row) + "\n")

    print(f"\n✅ Benchmark saved to: {results_file}")

if __name__ == "__main__":
    main()

