
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Path to the TFLite model file (must be in the same folder)
MODEL_PATH = "mobilenetv2_best_model.tflite"

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit UI
st.set_page_config(page_title="Defect Detection - MobileNetV2 (TFLite)", layout="wide")
st.title("🧪 AI Defect Detection App (TFLite)")

st.markdown("Upload an image to check whether it is **Good** or **Defective**.")

uploaded_file = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess
    input_data = preprocess_image(img)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]

    label = "✅ Good Product (0)" if prediction <= 0.5 else "⚠️ Defective Product (1)"
    confidence = 1 - prediction if prediction <= 0.5 else prediction

    st.subheader("Prediction:")
    st.write(label)
    st.write(f"Confidence: `{confidence:.4f}`")
