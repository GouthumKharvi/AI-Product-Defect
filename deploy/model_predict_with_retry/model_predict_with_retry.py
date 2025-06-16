<<<<<<< HEAD

import streamlit as st

def model_predict_with_retry(model, img_array, retries=1):
    for attempt in range(retries + 1):
        try:
            pred = model.predict(img_array)[0][0]
            conf = pred if pred > 0.5 else 1 - pred
            return pred, conf
        except Exception as e:
            st.error(f"Prediction failed on attempt {attempt + 1}: {e}")
            if attempt == retries:
                raise e
=======

import streamlit as st

def model_predict_with_retry(model, img_array, retries=1):
    for attempt in range(retries + 1):
        try:
            pred = model.predict(img_array)[0][0]
            conf = pred if pred > 0.5 else 1 - pred
            return pred, conf
        except Exception as e:
            st.error(f"Prediction failed on attempt {attempt + 1}: {e}")
            if attempt == retries:
                raise e
>>>>>>> 67bf31b040a6ab11782d8fbb67b4333cb2a2e60b
