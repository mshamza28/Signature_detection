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
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


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
    result = "Forged Signature" if class_idx == 1 else "Original Signature"

    # Display the image with prediction
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {result}')
    plt.show()
    return result
# Streamlit UI
st.title('Forged Signature Detetction')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting...")
    x = predict_and_show(uploaded_file)
    st.write(x)
