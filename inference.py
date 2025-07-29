import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from unet_autoencoder import UNetAutoencoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path):
    model = UNetAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def preprocess_image(image_file):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = Image.open(image_file).convert('L')
    return transform(img).unsqueeze(0)  # add batch dim

def compute_anomaly_map(original, reconstructed):
    error_map = torch.abs(original - reconstructed)
    error_map = error_map.squeeze().cpu().numpy()
    norm_error = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm_error

def apply_colormap(error_map):
    heatmap = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def overlay_heatmap_on_image(original_img, heatmap, alpha=0.5):
    if original_img.ndim == 2:
        original_img = np.stack([original_img]*3, axis=-1)  # grayscale to RGB
    overlayed = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return overlayed