# test_model_load.py
from tflite_runtime.interpreter import Interpreter, load_delegate
import numpy as np
import os

# --- ⚠️ UPDATE THIS PATH TO YOUR MODEL ---
MODEL_PATH = "models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite" # Use the same path as your main script

print(f"Attempting to load model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file does not exist at '{MODEL_PATH}'")
    exit()

# --- Test 1: Load on CPU ---
print("\n--- Testing on CPU ---")
try:
    interpreter_cpu = Interpreter(model_path=MODEL_PATH)
    interpreter_cpu.allocate_tensors()
    print("SUCCESS: Model loaded and tensors allocated on CPU.")
    input_details_cpu = interpreter_cpu.get_input_details()
    print(f"CPU Input details: {input_details_cpu[0]['shape']}, dtype: {input_details_cpu[0]['dtype']}")
except Exception as e:
    print(f"ERROR loading on CPU: {e}")

# --- Test 2: Attempt to load with EdgeTPU delegate ---
print("\n--- Testing with EdgeTPU Delegate ---")
try:
    delegates = [load_delegate('libedgetpu.so.1')]
    print("EdgeTPU delegate found.")
    interpreter_tpu = Interpreter(model_path=MODEL_PATH, experimental_delegates=delegates)
    interpreter_tpu.allocate_tensors()
    print("SUCCESS: Model loaded and tensors allocated with EdgeTPU delegate.")
    input_details_tpu = interpreter_tpu.get_input_details()
    print(f"TPU Input details: {input_details_tpu[0]['shape']}, dtype: {input_details_tpu[0]['dtype']}")
except Exception as e:
    print(f"ERROR loading with EdgeTPU delegate: {e}")

print("\nTest finished.")