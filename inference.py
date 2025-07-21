import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2  # for color mapping
import os

from unet_autoencoder import UNetAutoencoder  # your model file inside project_folder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path):
    model = UNetAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert('L')
    return transform(img).unsqueeze(0)  # add batch dim

def compute_anomaly_map(original, reconstructed):
    # Both tensors of shape [1,1,H,W]
    error_map = torch.abs(original - reconstructed)
    error_map = error_map.squeeze().cpu().numpy()

    # Normalize error map to 0-255 uint8
    norm_error = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm_error

def apply_colormap(error_map):
    # Apply a color map (JET) to the error map
    heatmap = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return heatmap

def overlay_heatmap_on_image(original_img, heatmap, alpha=0.5):
    # original_img: numpy array (H,W), grayscale [0-255]
    # heatmap: numpy array (H,W,3), color
    original_img = np.stack([original_img]*3, axis=-1)  # grayscale to 3 channels
    overlayed = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return overlayed

def run_inference_on_image(input_image_path):
    # Paths relative to project_folder, so add ../ to go one level up
    MODEL_PATH = '../unet_epoch_20.pth'  

    model = load_model(MODEL_PATH)
    input_tensor = preprocess_image(input_image_path).to(DEVICE)

    with torch.no_grad():
        reconstructed = model(input_tensor)

    original_np = (input_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    anomaly_map = compute_anomaly_map(input_tensor, reconstructed)

    heatmap = apply_colormap(anomaly_map)

    overlayed_img = overlay_heatmap_on_image(original_np, heatmap)

    return overlayed_img

def generate_output_image(input_path, output_path):
    anomaly_overlay = run_inference_on_image(input_path)
    output_img = Image.fromarray(anomaly_overlay)
    output_img.save(output_path)

if __name__ == '__main__':
    # For testing, relative path to image is one level up
    image_path = '../normal_xrays/Image_1.png'
    output_path = 'output_test.png'  # saves inside project_folder
    generate_output_image(image_path, output_path)
    print(f"Output saved to {output_path}")
