import requests
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from streamlit_lottie import st_lottie

# Load model with error handling
try:
    model = load_model('best_model.h5')
except Exception as e:
    st.error("Failed to load model: " + str(e))
    model = None

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

# Image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224), Image.LANCZOS)
    img = img.convert('RGB')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

# Prediction function
def predict_signature(img_data):
    x = preprocess_image(img_data)
    prediction = model.predict(x)
    return np.argmax(prediction[0])

# UI
st.title('🔍 Forged Signature Detection')
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload a signature image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔎 Verify Signature"):
        if model:
            st.write("⏳ **Verifying...**")
            
            # Load animation
            lottie_url = "https://lottie.host/48a98916-5ce1-41f7-a043-b5932bc5c542/w183dqaRuZ.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, height=200)
            
            result = predict_signature(uploaded_image)

            st.markdown("---")
            st.subheader("Result:")
            if result == 1:
                st.error("🚨 The Signature is **Forged** ❌")
            else:
                st.balloons()
                st.success("✅ The Signature is **Original** 🎉")
        else:
            st.error("Model is not loaded. Please check the file.")

