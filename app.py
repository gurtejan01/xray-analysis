import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
from inference import load_model, preprocess_image, compute_anomaly_map
from unet_autoencoder import UNetAutoencoder
import os

# Optional: Confirm OpenCV is working
st.write("OpenCV version:", cv2.__version__)

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
    st.image(image, caption="Original X-ray", use_container_width=True)

    # Preprocess and run inference
    input_tensor = preprocess_image(uploaded_file)
    input_tensor = input_tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        reconstructed = model(input_tensor)

    # Compute anomaly map
    anomaly_map = compute_anomaly_map(input_tensor, reconstructed)

    # Convert to numpy if needed
    if not isinstance(anomaly_map, np.ndarray):
        anomaly_map = anomaly_map.cpu().numpy()

    # Normalize anomaly map safely
    max_val = anomaly_map.max()
    if max_val > 0:
        anomaly_map = (anomaly_map * 255 / max_val).astype(np.uint8)
    else:
        anomaly_map = (anomaly_map * 255).astype(np.uint8)

    # Debug info
    st.write("Anomaly map shape:", anomaly_map.shape)
    st.write("Anomaly map dtype:", anomaly_map.dtype)
    st.write("Anomaly map min value:", anomaly_map.min())
    st.write("Anomaly map max value:", anomaly_map.max())
    st.write("Anomaly map mean value:", anomaly_map.mean())
    st.write("Anomaly map std deviation:", anomaly_map.std())

    # Show grayscale anomaly map (before heatmap)
    st.image(anomaly_map, caption="Normalized Anomaly Map (grayscale)", clamp=True, use_container_width=True)

    # Threshold anomaly map to highlight stronger anomalies
    threshold = 30  # you can adjust this threshold
    anomaly_map_thresh = np.where(anomaly_map > threshold, anomaly_map, 0).astype(np.uint8)

    st.image(anomaly_map_thresh, caption="Thresholded Anomaly Map", clamp=True, use_container_width=True)

    # Apply heatmap on thresholded map
    heatmap = cv2.applyColorMap(anomaly_map_thresh, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Display heatmap
    st.image(heatmap, caption="Thresholded Anomaly Heatmap", use_container_width=True)
