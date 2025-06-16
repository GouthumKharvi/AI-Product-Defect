
import streamlit as st
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from PIL import Image
import socket

st.set_page_config(page_title="Defect Detector - Deep Learning App", layout="wide")

MODEL_PATHS = {
    "MobileNetV2 🔁": r"C:\Users\Gouthum\Downloads\AI Projects CNN\models\mobilenetv2_best_model.h5",
    "Custom CNN 🧠": r"C:\Users\Gouthum\Downloads\AI Projects CNN\models\cnn_best_model.h5",
    "EfficientNetB0 ⚡": r"C:\Users\Gouthum\Downloads\AI Projects CNN\models\efficientnet_final_finetuned.h5"
}

LAST_CONV_LAYERS = {
    "MobileNetV2 🔁": "Conv_1",
    "Custom CNN 🧠": "conv2d_3",
    "EfficientNetB0 ⚡": "top_activation"
}

@st.cache_data
def load_trained_model(path):
    return load_model(path)

def is_healthy():
    return {"status": "UP", "host": socket.gethostname()}

def show_gradcam(img_array, model, last_conv_layer_name):
    import cv2

    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + K.epsilon())
    heatmap = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    original_img = (img_array[0] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

    st.image(overlay, caption="Grad-CAM Visualization", use_column_width=True)

# === SIDEBAR UI ===
st.sidebar.title("⚙️ Settings")
selected_model_name = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))
model = load_trained_model(MODEL_PATHS[selected_model_name])

if st.sidebar.button("🔍 Run Health Check"):
    health = is_healthy()
    st.sidebar.success(f"✅ {health['status']} - Host: {health['host']}")

st.sidebar.markdown("---")
st.sidebar.info("Use this panel to upload your image and choose your preferred model.")
st.sidebar.caption("Built with ❤️ using Streamlit and Keras")

# === MAIN HEADER ===
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color:#4CAF50;">🧪 AI Defect Detection App</h1>
        <p style="font-size: 18px;">Upload a product image to detect whether it is <b>Good</b> or <b>Defective</b> using deep learning models.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption="🖼️ Uploaded Image", width=250)
        with col2:
            st.markdown("### 🛠️ Processing your image...")

    img = img.resize((224, 224) if selected_model_name == "MobileNetV2 🔁" else (256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = " ⚠️ Defective product detected: not suitable for sale or use (1)" if prediction > 0.5 else "✅ High-quality product approved for delivery and customer use (0)"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    with st.container():
        st.markdown("---")
        st.markdown(f"""
            <div style="padding: 20px; background-color: ##121212; border-radius: 10px;">
                <h3>🔍 Prediction Result</h3>
                <p><b>Status:</b> {label}</p>
                <p><b>Model Used:</b> {selected_model_name}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📣 Feedback")
    feedback = st.radio("Was the prediction correct?", ["Yes", "No"], key="feedback")

    if st.button("Submit Feedback"):
        os.makedirs("logs", exist_ok=True)
        with open("logs/feedback.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now()} | Model: {selected_model_name} | Prediction: {label} | Feedback: {feedback}\n")
        st.success("✅ Feedback received successfully!")

    if st.button("🔍 Show Grad-CAM"):
        st.subheader("🧠 Grad-CAM Heatmap")
        show_gradcam(img_array, model, last_conv_layer_name=LAST_CONV_LAYERS[selected_model_name])

st.markdown("---")
st.markdown("""
    <div style='text-align: center; font-size: 14px; color: gray;'>
        © 2025 Defect Detector App | Developed By Gouthum 
    </div>
""", unsafe_allow_html=True)
