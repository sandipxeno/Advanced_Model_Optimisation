from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = '/Users/swedha/Documents/Advanced_Model_Optimisation/Flask/static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


UPLOAD_FOLDER = '/Users/swedha/Documents/Advanced_Model_Optimisation/Flask/static/uploads'
OUTPUT_FOLDER = '/Users/swedha/Documents/Advanced_Model_Optimisation/test(out_put)'  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO("/Users/swedha/Documents/Advanced_Model_Optimisation/models/yolo12n.quantized.onnx")

video_path = ""
detected_labels = set()

@app.route('/')
def index():
    return render_template('index.html', video_uploaded=False)

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path, detected_labels
    detected_labels.clear()

    if 'video' not in request.files:
        return "No file part"

    file = request.files['video']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        print(f"[INFO] Video uploaded to: {video_path}")
        return render_template('index.html', video_uploaded=True)

    return "Upload failed"

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_name = os.path.splitext(os.path.basename(video_path))[0] + "_output.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
