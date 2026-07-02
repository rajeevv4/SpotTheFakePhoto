import os
import sys
import time
import cv2
import threading
import numpy as np
from flask import Flask, render_template, Response, jsonify, send_file

# Add project root to python path to resolve src imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress debug logs
import logging
logging.disable(logging.WARNING)

from src.inference import SpotTheFakePicPredictor
from src.feature_extraction import extract_features_pipeline

app = Flask(__name__)

@app.route('/background.png')
def background_img():
    # Return the generated tech UI background asset
    bg_path = "/Users/rajeev/.gemini/antigravity-ide/brain/2026f22c-e324-4e94-9310-669eff6f15d7/web_ui_background_1782994849956.png"
    return send_file(bg_path, mimetype='image/png')

# Initialize Hot Predictor once at startup
model_path = os.path.join(project_root, "models", "model.pkl")
predictor = SpotTheFakePicPredictor(model_path=model_path)

class WebcamStream:
    """Thread-safe webcam capture and asynchronous model inference manager."""
    
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            # Fallback if primary input is unavailable
            for alt_src in [1, 2, -1]:
                self.stream = cv2.VideoCapture(alt_src)
                if self.stream.isOpened():
                    break
                    
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.latest_probability = 0.5
        self.inference_lock = threading.Lock()
        
    def start(self):
        if self.started:
            return self
        self.started = True
        
        # Frame capture thread
        self.capture_thread = threading.Thread(target=self._update_frames, args=())
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Prediction inference thread (runs decoupled from video rate)
        self.inference_thread = threading.Thread(target=self._run_inference, args=())
        self.inference_thread.daemon = True
        self.inference_thread.start()
        return self
        
    def _update_frames(self):
        while self.started:
            grabbed, frame = self.stream.read()
            if grabbed:
                # Mirror the frame horizontally for natural user display
                frame = cv2.flip(frame, 1)
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            time.sleep(0.01) # Limit capture loop overhead
            
    def read(self):
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy
            
    def _run_inference(self):
        while self.started:
            grabbed, frame = self.read()
            if grabbed and frame is not None:
                try:
                    # Run existing preprocess + feature extraction pipeline
                    features = extract_features_pipeline(frame)
                    features_reshaped = features.reshape(1, -1)
                    prob = predictor.pipeline.predict_proba(features_reshaped)[0][1]
                    with self.inference_lock:
                        self.latest_probability = float(prob)
                except Exception as e:
                    # Silent fallback in thread to prevent crashes
                    pass
            time.sleep(0.15) # Run prediction ~6-7 times per second (low CPU usage)
            
    def stop(self):
        self.started = False
        self.stream.release()

# Global stream instance
webcam = None

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global webcam
    while True:
        success, frame = webcam.read()
        if not success or frame is None:
            # Send placeholder offline frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8) + 40
            cv2.putText(placeholder, "No Webcam Signal Found", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.2)
            continue
            
        # Downscale stream slightly for network efficiency
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # Cap streaming frame rate at ~30 FPS

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global webcam
    prob = 0.5
    if webcam is not None:
        with webcam.inference_lock:
            prob = webcam.latest_probability
            
    prediction = "SCREEN" if prob >= 0.5 else "REAL"
    color = "red" if prob >= 0.5 else "green"
    
    return jsonify({
        "probability": round(prob, 4),
        "prediction": prediction,
        "color": color
    })

if __name__ == '__main__':
    # Initialize and start webcam capture
    webcam = WebcamStream().start()
    print("--------------------------------------------------")
    print("  Spot The Fake Photo - Live Webcam Demo Server   ")
    print("  URL: http://127.0.0.1:5000                      ")
    print("  Press Ctrl+C in terminal to stop.               ")
    print("--------------------------------------------------")
    try:
        app.run(host='127.0.0.1', port=5000, debug=False)
    finally:
        webcam.stop()
