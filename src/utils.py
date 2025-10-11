# src/utils.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def preprocess_image(image, target_size=224):
    """
    Preprocess image for ViT model.
    
    Args:
        image: PIL Image or file path
        target_size: Target size for resizing
    
    Returns:
        PIL.Image: Preprocessed image
    """
    if isinstance(image, str):
        # If it's a file path, load the image
        image = Image.open(image)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize((target_size, target_size))
    
    return image

def normalize_heatmap(heatmap):
    """
    Normalize heatmap to [0, 1] range.
    
    Args:
        heatmap: numpy array of heatmap values
    
    Returns:
        numpy.array: Normalized heatmap
    """
    if heatmap.max() > heatmap.min():
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        return np.zeros_like(heatmap)

def overlay_heatmap(image, heatmap, alpha=0.5, colormap='hot'):
    """
    Overlay heatmap on original image.
    
    Args:
        image: PIL Image
        heatmap: numpy array of heatmap values
        alpha: Transparency for heatmap overlay
        colormap: Matplotlib colormap name
    
    Returns:
        PIL.Image: Image with heatmap overlay
    """
    # Normalize heatmap
    heatmap = normalize_heatmap(heatmap)
    
    # Convert heatmap to RGB using colormap
    cmap = plt.get_cmap(colormap)
    heatmap_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    
    # Resize heatmap to match image size
    heatmap_img = Image.fromarray(heatmap_rgb)
    heatmap_img = heatmap_img.resize(image.size, Image.Resampling.LANCZOS)
    
    # Blend images
    original_rgba = image.convert('RGBA')
    heatmap_rgba = heatmap_img.convert('RGBA')
    blended = Image.blend(original_rgba, heatmap_rgba, alpha)
    
    return blended.convert('RGB')

def create_comparison_figure(original_image, explanation_images, explanation_titles):
    """
    Create a comparison figure showing original image and multiple explanations.
    
    Args:
        original_image: PIL Image
        explanation_images: List of explanation images
        explanation_titles: List of titles for each explanation
    
    Returns:
        matplotlib.figure.Figure: Comparison figure
    """
    num_explanations = len(explanation_images)
    fig, axes = plt.subplots(1, num_explanations + 1, figsize=(4 * (num_explanations + 1), 4))
    
    # Plot original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Plot explanations
    for i, (exp_img, title) in enumerate(zip(explanation_images, explanation_titles)):
        axes[i + 1].imshow(exp_img)
        axes[i + 1].set_title(title, fontweight='bold')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    return fig

def tensor_to_image(tensor):
    """
    Convert PyTorch tensor to PIL Image.
    
    Args:
        tensor: PyTorch tensor of shape (C, H, W) or (B, C, H, W)
    
    Returns:
        PIL.Image: Converted image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize if needed and convert to numpy
    tensor = tensor.cpu().detach()
    if tensor.min() < 0 or tensor.max() > 1:
        # Assume it's normalized, denormalize to [0, 1]
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    numpy_image = tensor.permute(1, 2, 0).numpy()
    numpy_image = (numpy_image * 255).astype(np.uint8)
    
    return Image.fromarray(numpy_image)

def get_top_predictions_dict(probs, labels, top_k=5):
    """
    Convert top predictions to dictionary for Gradio Label component.
    
    Args:
        probs: Array of probabilities
        labels: List of label names
        top_k: Number of top predictions to include
    
    Returns:
        dict: Dictionary of {label: probability} for top-k predictions
    """
    return {label: float(prob) for label, prob in zip(labels[:top_k], probs[:top_k])}