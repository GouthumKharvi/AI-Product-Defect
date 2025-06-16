
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
MODEL_PATH = "mobilenetv2_best_model.h5"
model = load_model(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="Defect Detection - MobileNetV2", layout="wide")
st.title("🧪 AI Defect Detection App")
st.markdown("Upload an image to check whether it is **Good** or **Defective**.")

uploaded_file = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "✅ Good Product (0)" if prediction <= 0.5 else "⚠️ Defective Product (1)"
    confidence = 1 - prediction if prediction <= 0.5 else prediction

    st.subheader("Prediction:")
    st.write(label)
    st.write(f"Confidence: `{confidence:.4f}`")
