import cv2
import numpy as np
import threading
from flask import Flask, Response
from tflite_runtime.interpreter import Interpreter, load_delegate
import time

# ------------------------------------------------------------------------------------
# --- ⚙️ CONFIGURATION - UPDATE THESE VALUES AS PER YOUR SETUP ⚙️ ---
# ------------------------------------------------------------------------------------
RTSP_URL = "rtsp://admin:Sasi2009@192.168.29.117:554/Streaming/Channels/101"  # ⚠️ UPDATE THIS!
MODEL_PATH = "models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"  # ⚠️ UPDATE THIS! Path to your .tflite model
                                             # Example: "/home/pi/models/ssd_mobilenet_v2_coco_quant_postprocessed_edgetpu.tflite"
                                             # or "/home/pi/models/ssd_mobilenet_v2.tflite"

# Attempt to load EdgeTPU delegate, if not available, will run on CPU
# For this to work, MODEL_PATH should ideally be an EdgeTPU compiled model
# And you must have the EdgeTPU runtime installed.
USE_EDGETPU = True # Set to False to force CPU TFLite execution

# COCO Labels (SSD MobileNet V2 is often trained on COCO)
# The output class IDs are 0-indexed into this list.
# 'person' is typically the first class (index 0) if background is not class 0.
# Some models might have 'background' as class 0, then 'person' as class 1.
# Check your specific model's documentation if unsure. We'll assume 'person' is ID 0 here.
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
PERSON_LABEL_INDEX = COCO_LABELS.index("person") # Should be 0 if the list is correct for your model

CONFIDENCE_THRESHOLD = 0.5  # Minimum score to consider a detection
INPUT_IMG_WIDTH = 300       # Expected input width for SSD MobileNet V2
INPUT_IMG_HEIGHT = 300      # Expected input height for SSD MobileNet V2

# Flask server settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
# ------------------------------------------------------------------------------------
# --- END OF CONFIGURATION ---
# ------------------------------------------------------------------------------------

# --- Global variables ---
output_frame = None
frame_lock = threading.Lock()
interpreter = None
input_details = None
output_details = None
is_model_quantized_uint8 = False # Will be determined during model load

def load_model():
    """Loads the TFLite model and prepares the interpreter."""
    global interpreter, input_details, output_details, USE_EDGETPU, is_model_quantized_uint8

    delegates_list = []
    if USE_EDGETPU:
        try:
            print("Attempting to load EdgeTPU delegate...")
            delegates_list.append(load_delegate('libedgetpu.so.1'))
            print("EdgeTPU delegate loaded successfully.")
        except (ValueError, OSError) as e:
            print(f"Error loading EdgeTPU delegate: {e}")
            print("Continuing with CPU execution for TFLite model.")
            USE_EDGETPU = False # Fallback to CPU if delegate fails
    else:
        print("EdgeTPU not selected. Using CPU for TFLite model.")


    try:
        print(f"Loading model from: {MODEL_PATH}")
        interpreter = Interpreter(model_path=MODEL_PATH, experimental_delegates=delegates_list if delegates_list else None)
        interpreter.allocate_tensors()
        print("Model loaded and tensors allocated.")

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Check if the model is quantized (uint8 input)
        if input_details[0]['dtype'] == np.uint8:
            is_model_quantized_uint8 = True
            print("Model expects uint8 input (quantized).")
        else:
            is_model_quantized_uint8 = False
            print("Model expects float32 input.")

        # Print some model details (optional)
        print(f"  Input tensor: {input_details[0]['name']}, Shape: {input_details[0]['shape']}, Type: {input_details[0]['dtype']}")
        # print(f"  Output tensor details: {output_details}")


    except Exception as e:
        print(f"CRITICAL: Failed to load TFLite model or allocate tensors: {e}")
        interpreter = None # Ensure interpreter is None if loading fails

def capture_and_detect_persons():
    """Captures frames from RTSP and performs person detection."""
    global output_frame, frame_lock, interpreter, input_details, output_details

    if interpreter is None:
        print("Model not loaded. Detection thread cannot start.")
        # Create a placeholder frame indicating model error
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Error: Model not loaded", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        with frame_lock:
            output_frame = placeholder.copy()
        return

    print(f"Attempting to open RTSP stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream: {RTSP_URL}")
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Error: Cannot open RTSP", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        with frame_lock:
            output_frame = placeholder.copy()
        return

    print("RTSP stream opened successfully. Starting detection...")
    frame_count = 0
    start_time = time.time()
    model_input_height = input_details[0]['shape'][1]
    model_input_width = input_details[0]['shape'][2]

    while True:
        success, frame_original = cap.read()
        if not success:
            print("Warning: Failed to grab frame from RTSP stream. Attempting to reconnect...")
            cap.release()
            time.sleep(2) # Wait a bit
            cap = cv2.VideoCapture(RTSP_URL)
            if not cap.isOpened():
                print("Failed to reconnect to RTSP. Exiting detection thread.")
                break
            print("Reconnected to RTSP stream.")
            continue

        frame_rgb = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB) # TFLite models often expect RGB
        frame_resized = cv2.resize(frame_rgb, (model_input_width, model_input_height))
        input_data = np.expand_dims(frame_resized, axis=0) # Add batch dimension

        if not is_model_quantized_uint8: # If float model
            input_data = (np.float32(input_data) - 127.5) / 127.5 # Normalize to [-1, 1] - common for some float models

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Output tensor structure for SSD MobileNet V2 typically:
        # output_details[0]: detection_boxes (normalized ymin, xmin, ymax, xmax)
        # output_details[1]: detection_classes (0-indexed class IDs)
        # output_details[2]: detection_scores
        # output_details[3]: num_detections
        # The exact order might vary, check your model's output_details if issues arise.
        # For models from TF Object Detection API, they are often in this order.

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class IDs
        scores = interpreter.get_tensor(output_details[2]['index'])[0]   # Confidence scores
        # num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0]) # Number of detections

        frame_annotated = frame_original.copy()
        frame_height_orig, frame_width_orig, _ = frame_annotated.shape

        for i in range(len(scores)): # TFOD API models might not use num_detections directly
            if scores[i] >= CONFIDENCE_THRESHOLD and int(classes[i]) == PERSON_LABEL_INDEX:
                ymin, xmin, ymax, xmax = boxes[i]
                # Convert normalized coordinates to pixel values
                left = int(xmin * frame_width_orig)
                top = int(ymin * frame_height_orig)
                right = int(xmax * frame_width_orig)
                bottom = int(ymax * frame_height_orig)

                cv2.rectangle(frame_annotated, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{COCO_LABELS[int(classes[i])]}: {scores[i]:.2f}"
                cv2.putText(frame_annotated, label, (left, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            fps_text = f"FPS: {fps:.2f} ({'EdgeTPU' if USE_EDGETPU else 'CPU'})"
            cv2.putText(frame_annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            frame_count = 0
            start_time = time.time()
        else:
            cv2.putText(frame_annotated, f"FPS: Calculating... ({'EdgeTPU' if USE_EDGETPU else 'CPU'})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        with frame_lock:
            global output_frame
            output_frame = frame_annotated.copy()

        time.sleep(0.001) # Minimal sleep to prevent busy-waiting if processing is very fast

    cap.release()
    print("Detection thread finished.")

def generate_frames_for_stream():
    """Generates JPEG frames for video streaming."""
    frame_width, frame_height = 640, 480 # Default placeholder size
    while True:
        time.sleep(1/25) # Aim for ~25 FPS for the output stream
        with frame_lock:
            if output_frame is None:
                placeholder = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Initializing stream...", (frame_width // 2 - 150, frame_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
                (flag, encoded_image) = cv2.imencode(".jpg", placeholder)
                if not flag: continue
            else:
                if output_frame.shape[0] != frame_height or output_frame.shape[1] != frame_width :
                    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
                if not flag: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return f"""
    <html>
        <head><title>SSD MobileNet V2 Person Detection</title></head>
        <body>
            <h1>SSD MobileNet V2 Person Detection ({'EdgeTPU' if USE_EDGETPU else 'CPU'})</h1>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed_route():
    return Response(generate_frames_for_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("--- SSD MobileNet V2 Person Detection Server ---")
    print("Initializing TFLite model...")
    load_model() # Load the model

    if interpreter is not None:
        print("Starting RTSP capture and detection thread...")
        detection_thread = threading.Thread(target=capture_and_detect_persons, daemon=True)
        detection_thread.start()
    else:
        print("Failed to load model. Flask server will run but stream may show errors.")

    print(f"Starting Flask server... Open http://<YOUR_PI_IP>:{FLASK_PORT} in your browser.")
    if RTSP_URL == "YOUR_RTSP_STREAM_URL_HERE" or MODEL_PATH == "YOUR_SSD_MOBILENET_V2.tflite":
        print("\n⚠️⚠️⚠️ WARNING: RTSP_URL or MODEL_PATH is not set! Please update the script. ⚠️⚠️⚠️\n")

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)