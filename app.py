import streamlit as st
from PIL import Image
import torch
import numpy as np
from inference import (
    load_model,
    preprocess_image,
    compute_anomaly_map,
    apply_colormap,
    overlay_heatmap_on_image
)

st.title("Chest X-ray Anomaly Detector")
st.write("Upload a chest X-ray to see potential anomalies highlighted using a UNet Autoencoder.")

@st.cache_resource
def load_trained_model():
    return load_model('unet_epoch_30.pth')

model = load_trained_model()

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display original grayscale image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Original X-ray", use_container_width=True)

    # Preprocess and run inference
    input_tensor = preprocess_image(uploaded_file).to(next(model.parameters()).device)
    with torch.no_grad():
        reconstructed = model(input_tensor)

    # Generate anomaly heatmap overlay
    anomaly_map = compute_anomaly_map(input_tensor, reconstructed)
    heatmap = apply_colormap(anomaly_map)
    original_np = (input_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    overlay = overlay_heatmap_on_image(original_np, heatmap)

    # Show final result
    st.image(overlay, caption="Anomaly Heatmap Overlay", use_container_width=True)
