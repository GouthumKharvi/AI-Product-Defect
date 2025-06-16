
import os
import datetime

def log_low_confidence(prediction, confidence, model_name=None, image_filename=None):
    os.makedirs("logs", exist_ok=True)
    log_file_path = os.path.join("logs", "low_confidence.log")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "Defective" if prediction > 0.5 else "Good"

    log_entry = (
        f"{timestamp} | Model: {model_name} | Status: {status} | "
        f"Confidence: {confidence:.4f} | Image: {image_filename or 'N/A'}\n"
    )

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_entry)

    print("📄 Low confidence entry logged.")
