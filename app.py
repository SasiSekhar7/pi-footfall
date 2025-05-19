# tpu_detector_server.py
import cv2
import numpy as np
import threading
from flask import Flask, Response, jsonify
from tflite_runtime.interpreter import Interpreter, load_delegate
import time
import os

# ------------------------------------------------------------------------------------
# --- ⚙️ CONFIGURATION - YOU MUST UPDATE THESE! ⚙️ ---
# ------------------------------------------------------------------------------------
# ⚠️ 1. UPDATE THIS WITH YOUR RTSP STREAM URL ⚠️
RTSP_URL = "rtsp://admin:Sasi2009@192.168.29.117:554/Streaming/Channels/101"

# ⚠️ 2. UPDATE THIS WITH THE FULL PATH TO YOUR *EdgeTPU-COMPILED* TFLITE MODEL FILE ⚠️
MODEL_PATH = "models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite" # Example: "/home/pi/models/ssd_mobilenet_v2_coco_quant_postprocessed_edgetpu.tflite"

# --- EdgeTPU Configuration ---
# Attempt to use EdgeTPU by default.
# For this to work:
#   1. MODEL_PATH MUST be an EdgeTPU-compiled model.
#   2. EdgeTPU runtime (libedgetpu.so.1 and correct tflite-runtime) MUST be installed.
USE_EDGETPU = False

# --- Model and Detection Parameters ---
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
PERSON_LABEL_NAME = "person"
try:
    PERSON_LABEL_INDEX = COCO_LABELS.index(PERSON_LABEL_NAME)
except ValueError:
    print(f"Error: '{PERSON_LABEL_NAME}' not found in COCO_LABELS. Defaulting to index 0.")
    PERSON_LABEL_INDEX = 0

CONFIDENCE_THRESHOLD = 0.4
INPUT_IMG_WIDTH = 300
INPUT_IMG_HEIGHT = 300

# --- Flask Server Settings ---
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
# ------------------------------------------------------------------------------------
# --- END OF CONFIGURATION ---
# ------------------------------------------------------------------------------------

output_frame_global = None
frame_lock_global = threading.Lock()
tflite_interpreter_global = None
tflite_input_details_global = None
tflite_output_details_global = None
model_quantized_uint8_global = False
model_status_message_global = "Model not loaded."
active_device_global = "CPU (Fallback)" # Assume CPU until TPU is confirmed

def load_tflite_model():
    global tflite_interpreter_global, tflite_input_details_global, tflite_output_details_global
    global model_quantized_uint8_global, model_status_message_global, USE_EDGETPU, active_device_global

    model_status_message_global = f"Loading model: {os.path.basename(MODEL_PATH)}..."
    print(model_status_message_global)

    if not os.path.exists(MODEL_PATH):
        model_status_message_global = f"ERROR: Model file not found at {MODEL_PATH}"
        print(model_status_message_global)
        active_device_global = "Error: Model File Missing"
        return

    delegates_list = []
    if USE_EDGETPU:
        try:
            print("Attempting to load EdgeTPU delegate ('libedgetpu.so.1')...")
            delegates_list.append(load_delegate('libedgetpu.so.1'))
            active_device_global = "EdgeTPU"
            print("EdgeTPU delegate loaded successfully.")
        except (ValueError, OSError) as e:
            print(f"Warning: Error loading EdgeTPU delegate: {e}")
            print("Falling back to CPU for TFLite model.")
            active_device_global = "CPU (EdgeTPU delegate failed)"
            # USE_EDGETPU = False # No need to change global USE_EDGETPU, just don't use delegate
    else:
        active_device_global = "CPU (TPU not selected)"
        print("EdgeTPU not selected by configuration. Using CPU for TFLite model.")

    try:
        tflite_interpreter_global = Interpreter(model_path=MODEL_PATH, experimental_delegates=delegates_list if active_device_global == "EdgeTPU" else None)
        tflite_interpreter_global.allocate_tensors()
        model_status_message_global = f"Model loaded ({active_device_global})"
        print(model_status_message_global)

        tflite_input_details_global = tflite_interpreter_global.get_input_details()
        tflite_output_details_global = tflite_interpreter_global.get_output_details()

        if tflite_input_details_global[0]['dtype'] == np.uint8:
            model_quantized_uint8_global = True
            print("Model expects uint8 input (quantized).")
        else:
            model_quantized_uint8_global = False
            print("Model expects float32 input.")
    except Exception as e:
        model_status_message_global = f"ERROR: Failed to load TFLite model or allocate tensors: {e}"
        print(model_status_message_global)
        active_device_global = f"Error: Model Load Fail ({active_device_global})"
        tflite_interpreter_global = None

def capture_and_detect_persons():
    global output_frame_global, frame_lock_global, tflite_interpreter_global
    global tflite_input_details_global, tflite_output_details_global, model_status_message_global, active_device_global

    if tflite_interpreter_global is None:
        model_status_message_global = "Model not loaded. Detection thread stopped."
        print(model_status_message_global)
        placeholder = np.zeros((INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Error: Model not loaded", (30, INPUT_IMG_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        with frame_lock_global: output_frame_global = placeholder.copy()
        return

    print(f"Attempting to open RTSP stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        model_status_message_global = f"Error: Could not open RTSP: {RTSP_URL}"
        print(model_status_message_global)
        placeholder = np.zeros((INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Error: Cannot open RTSP", (30, INPUT_IMG_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        with frame_lock_global: output_frame_global = placeholder.copy()
        return

    model_status_message_global = f"RTSP stream opened. Detecting on {active_device_global}..."
    print(model_status_message_global)

    frame_count = 0
    fps_calc_start_time = time.time()
    model_input_height = tflite_input_details_global[0]['shape'][1]
    model_input_width = tflite_input_details_global[0]['shape'][2]

    while True:
        success, frame_original = cap.read()
        if not success:
            model_status_message_global = "RTSP: Frame grab failed. Reconnecting..."
            print(model_status_message_global)
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            if not cap.isOpened():
                model_status_message_global = "RTSP: Reconnect failed. Stopping."
                print(model_status_message_global)
                break
            model_status_message_global = f"RTSP: Reconnected. Detecting on {active_device_global}..."
            print(model_status_message_global)
            continue

        frame_rgb = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (model_input_width, model_input_height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if not model_quantized_uint8_global:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        try:
            tflite_interpreter_global.set_tensor(tflite_input_details_global[0]['index'], input_data)
            tflite_interpreter_global.invoke()

            boxes = tflite_interpreter_global.get_tensor(tflite_output_details_global[0]['index'])[0]
            classes = tflite_interpreter_global.get_tensor(tflite_output_details_global[1]['index'])[0]
            scores = tflite_interpreter_global.get_tensor(tflite_output_details_global[2]['index'])[0]
        except Exception as e:
            print(f"Error during inference: {e}")
            # Potentially put error message on frame
            time.sleep(0.1) # avoid fast error loop
            continue


        frame_annotated = frame_original.copy()
        frame_height_orig, frame_width_orig, _ = frame_annotated.shape

        for i in range(len(scores)):
            if scores[i] >= CONFIDENCE_THRESHOLD and int(classes[i]) == PERSON_LABEL_INDEX:
                ymin, xmin, ymax, xmax = boxes[i]
                left, top = int(xmin * frame_width_orig), int(ymin * frame_height_orig)
                right, bottom = int(xmax * frame_width_orig), int(ymax * frame_height_orig)
                cv2.rectangle(frame_annotated, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{COCO_LABELS[PERSON_LABEL_INDEX]}: {scores[i]:.2f}"
                cv2.putText(frame_annotated, label, (left, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame_count += 1
        if frame_count >= 10:
            fps = frame_count / (time.time() - fps_calc_start_time)
            fps_text = f"FPS: {fps:.1f} ({active_device_global})"
            cv2.putText(frame_annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frame_count = 0
            fps_calc_start_time = time.time()
        else:
             cv2.putText(frame_annotated, f"FPS: ... ({active_device_global})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        with frame_lock_global:
            output_frame_global = frame_annotated.copy()
        time.sleep(0.001)

    cap.release()
    model_status_message_global = "Detection thread finished."
    print(model_status_message_global)

def generate_frames_for_stream():
    frame_width, frame_height = 640, 480
    while True:
        time.sleep(1/30) # Stream FPS
        with frame_lock_global:
            current_frame_to_encode = output_frame_global
        
        if current_frame_to_encode is None:
            placeholder = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            text_to_show = model_status_message_global if "Error" in model_status_message_global else "Initializing..."
            (text_width, text_height), _ = cv2.getTextSize(text_to_show, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            cv2.putText(placeholder, text_to_show, (frame_width//2 - text_width//2, frame_height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
            (flag, encoded_image) = cv2.imencode(".jpg", placeholder)
        else:
            if current_frame_to_encode.shape[0] != frame_height or current_frame_to_encode.shape[1] != frame_width :
                frame_height, frame_width = current_frame_to_encode.shape[0], current_frame_to_encode.shape[1]
            (flag, encoded_image) = cv2.imencode(".jpg", current_frame_to_encode)

        if not flag: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

app = Flask(__name__)

@app.route('/')
def index():
    return f"""
    <html> <head> <title>Person Detection ({active_device_global})</title>
            <style> body {{ font-family: sans-serif; }}
                    #status {{ margin-top: 10px; padding: 5px; background-color: #f0f0f0; border-radius: 5px; }}
            </style>
        </head>
        <body> <h1>Person Detection Stream</h1>
            <img src="/video_feed" width="640" height="480">
            <div id="status">Status: <span id="model_status_val">Loading...</span></div>
            <script>
                function fetchStatus() {{
                    fetch('/status')
                        .then(response => response.json())
                        .then(data => {{
                            document.getElementById('model_status_val').textContent = data.status;
                        }})
                        .catch(error => console.error('Error fetching status:', error));
                }}
                setInterval(fetchStatus, 2000); // Update status every 2 seconds
                fetchStatus(); // Initial call
            </script>
        </body> </html>"""

@app.route('/video_feed')
def video_feed_route():
    return Response(generate_frames_for_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status_route():
    with frame_lock_global: # Ensure consistent read
        status = model_status_message_global
    return jsonify({"status": status})


if __name__ == '__main__':
    print("--- Person Detection Server (Attempting EdgeTPU) ---")
    load_tflite_model()

    if tflite_interpreter_global is not None:
        print("Starting RTSP capture and detection thread...")
        detection_thread = threading.Thread(target=capture_and_detect_persons, daemon=True)
        detection_thread.start()
    else:
        print("CRITICAL: Model not loaded. Detection thread will not start.")
        # Ensure a status is available if model load fails before thread starts
        with frame_lock_global: output_frame_global = None # Ensure placeholder is used in generate_frames

    print(f"Flask server starting. Open http://<YOUR_PI_IP>:{FLASK_PORT} in your browser.")
    if RTSP_URL == "YOUR_RTSP_STREAM_URL_HERE" or MODEL_PATH == "YOUR_SSD_MOBILENET_V2_EDGETPU.tflite":
        print("\n⚠️⚠️⚠️ WARNING: RTSP_URL or MODEL_PATH might be using placeholder values! Please update the script. ⚠️⚠️⚠️\n")
    if USE_EDGETPU:
        print("INFO: Configured to USE_EDGETPU. Ensure MODEL_PATH is an EdgeTPU-compiled model and runtime is installed.")

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)