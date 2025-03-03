import requests
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from streamlit_lottie import st_lottie

# Load model with error handling
try:
    model = load_model('best_model.h5')
except Exception as e:
    st.error("🚨 Failed to load model. Please check your file: " + str(e))
    model = None

# Load Lottie animation
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
    class_idx = np.argmax(prediction[0])
    return class_idx

# UI Header
st.set_page_config(page_title="Signature Authentication", page_icon="✍️", layout="centered")
st.title('✍️ Forged Signature Detection')
st.markdown("**Upload a signature image or capture one from your webcam!** 🔍")

st.sidebar.title("📌 Instructions")
st.sidebar.write("1️⃣ Upload a signature image or use the webcam.")
st.sidebar.write("2️⃣ Click on **Verify Signature** to analyze authenticity.")
st.sidebar.write("3️⃣ See the result instantly.")

# File Upload Option
uploaded_file = st.file_uploader("📤 Upload Signature Image:", type=["jpg", "jpeg", "png"])

# Webcam Capture Option
st.markdown("---")
st.subheader("📷 Capture Signature Using Webcam")
captured_image = st.camera_input("Click below to capture")

# Determine the source of the image (uploaded or captured)
image_source = None
if uploaded_file:
    image_source = Image.open(uploaded_file)
elif captured_image:
    image_source = Image.open(captured_image)

if image_source:
    # Display uploaded/captured image
    st.image(image_source, caption="Selected Signature", use_column_width=True)

    # Grayscale transformation for visualization
    gray_image = ImageOps.grayscale(image_source)

    # Processing and Prediction
    if st.button("🔎 Verify Signature"):
        if model:
            st.write("⏳ **Processing...**")

            # Load animation
            lottie_url = "https://lottie.host/48a98916-5ce1-41f7-a043-b5932bc5c542/w183dqaRuZ.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, height=200)

            result = predict_signature(image_source)

            # Display result
            st.markdown("---")
            st.subheader("🔍 Result:")
            if result == 1:
                st.error("🚨 The Signature is **Forged** ❌")
            else:
                st.balloons()
                st.success("✅ The Signature is **Original** 🎉")

            # Show processed grayscale image
            st.markdown("---")
            st.subheader("📷 Processed Image View")
            fig, ax = plt.subplots()
            ax.imshow(gray_image, cmap="gray")
            ax.axis("off")
            st.pyplot(fig)

        else:
            st.error("⚠️ Model is not loaded. Please check your file.")

st.markdown("---")
st.write("💡 Developed with ❤️ using Streamlit & TensorFlow.")
