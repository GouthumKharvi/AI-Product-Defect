# monitoring_logs.py
import os
from datetime import datetime

# ✅ Define log folder path
LOG_DIR = os.path.join(os.path.dirname(__file__), "monitoring_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ Logging Functions
def log_uptime(event="App Started"):
    with open(os.path.join(LOG_DIR, "uptime.log"), "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | {event}\n")

def log_latency(seconds):
    with open(os.path.join(LOG_DIR, "latency.log"), "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | {seconds}\n")

def log_confidence(confidence):
    with open(os.path.join(LOG_DIR, "confidence.log"), "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | {confidence}\n")

def log_misclassification(label, confidence, image_array):
    with open(os.path.join(LOG_DIR, "misclassifications.log"), "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | Label: {label} | Confidence: {confidence} | ImageData: {str(image_array)[:200]}...\n")
