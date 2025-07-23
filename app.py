import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
from inference import load_model, preprocess_image, compute_anomaly_map
from unet_autoencoder import UNetAutoencoder
import os

# Load the model only once
@st.cache_resource
def load_trained_model():
    model_path = 'unet_epoch_30.pth'
    return load_model(model_path)

model = load_trained_model()

st.title("Chest X-ray Anomaly Detector")
st.write("Upload a chest X-ray and view the anomaly heatmap using our UNet Autoencoder.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Original X-ray", use_column_width=True)

    # Preprocess and run inference
    input_tensor = preprocess_image(uploaded_file)
    input_tensor = input_tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        reconstructed = model(input_tensor)

    # Compute anomaly map
    anomaly_map = compute_anomaly_map(input_tensor, reconstructed)

    # Normalize and convert anomaly map to heatmap
    anomaly_map = (anomaly_map * 255 / anomaly_map.max()).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    st.image(heatmap, caption="Anomaly Heatmap", use_column_width=True)
