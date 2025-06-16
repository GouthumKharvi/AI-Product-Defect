
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="Defect Detector - Deep Learning App", layout="wide")

MODEL_PATHS = {
    "MobileNetV2 🔁": "mobilenetv2_best_model.h5",
    "Custom CNN 🧠": "cnn_best_model.h5",
    "EfficientNetB0 ⚡": "efficientnet_final_finetuned.h5"
}

@st.cache_data
def load_trained_model(path):
    return load_model(path)

st.sidebar.title("⚙️ Settings")
selected_model_name = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))
model = load_trained_model(MODEL_PATHS[selected_model_name])
st.sidebar.markdown("---")
st.sidebar.info("Use this panel to upload your image and choose your preferred model.")
st.sidebar.caption("Built with ❤️ using Streamlit and Keras")

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
            <div style="padding: 20px; background-color: #121212; border-radius: 10px; color: white;">
                <h3>🔍 Prediction Result</h3>
                <p><b>Status:</b> {label}</p>
                <p><b>Model Used:</b> {selected_model_name}</p>
                <p><b>Confidence:</b> {confidence:.4f}</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div style='text-align: center; font-size: 14px; color: gray;'>
        © 2025 Defect Detector App | Developed By Gouthum 
    </div>
""", unsafe_allow_html=True)
