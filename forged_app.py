# -*- coding: utf-8 -*-
"""Forged_app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nuhBxN0Fx-lbN3JVQ_P_rkonzQloHBxv
"""

# Necessary Libraries
import requests
import streamlit as st
from PIL import Image
import numpy as np
import plotly.express as px
from io import BytesIO
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

# Theme Selection
def load_css(theme):
    if theme == 'dark':
        css = """
        <style>
            body {
                color: white;
                background-color: #333;
            }
            .big-font {
                font-size:20px !important;
                color: #FF4B4B;
            }
            .stButton>button {
                color: white;
                background-color: #FF4B4B;
            }
        </style>
        """
    else:
        css = """
        <style>
            body {
                color: black;
                background-color: #FFF;
            }
            .big-font {
                font-size:20px !important;
                color: #0078AA;
            }
            .stButton>button {
                color: black;
                background-color: #0078AA;
            }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# Initialize the theme based on user's previous preference or default to light
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

# Toggle theme function
def toggle_theme():
    if st.session_state['theme'] == 'light':
        st.session_state['theme'] = 'dark'
    else:
        st.session_state['theme'] = 'light'
    load_css(st.session_state['theme'])

# Sidebar or main page toggle
st.sidebar.button("Toggle Theme", on_click=toggle_theme)

# Load the initial or current theme CSS
load_css(st.session_state['theme'])


# Load Lottie
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()

# Load the pre-trained custom model
model = load_model('best_model.h5')

# Function to make predictions and display the image
def predict_and_show(img_data):
    img = Image.open(img_data)
    img = img.resize((224, 224), Image.ANTIALIAS)  # Resize image to match
    img = img.convert('RGB')  # Ensure image is RGB even if it was grayscale
    x = image.img_to_array(img)  # Convert the image to array
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)  # Preprocess the image
    prediction = model.predict(x)
    class_idx = np.argmax(prediction[0])
    return class_idx
    
# Streamlit UI
st.title('Forged Signature Detection')
st.markdown('---')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.markdown('---')
    st.write("")
    
    x = predict_and_show(uploaded_file)
    
    if st.button('Detect'):
        st.write("<p class='big-font'>Detecting...</p>", unsafe_allow_html=True)
        
        # Display Lottie animation
        lottie_url = "https://lottie.host/48a98916-5ce1-41f7-a043-b5932bc5c542/w183dqaRuZ.json"  # Lottie animation URL
        lottie_animation = load_lottieurl(lottie_url)
        st_lottie(lottie_animation, height=200, key="classification")

        # Print Result
        st.title('Result :')
        if x == 1:
            st.header("Forged Signature")
        else:
            st.header("Original Signature")
            
st.markdown('---')
