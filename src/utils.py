"""
Utility Functions Module

This module provides helper functions for image preprocessing, heatmap manipulation,
visualization, and data conversion used throughout the ViT auditing toolkit.

Author: ViT-XAI-Dashboard Team
License: MIT
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def preprocess_image(image, target_size=224):
    """
    Preprocess an image for Vision Transformer model input.

    This function handles loading images from file paths, converts them to RGB format,
    and resizes them to the target dimensions required by ViT models.

    Args:
        image (PIL.Image or str): Input image as a PIL Image object or file path string.
        target_size (int, optional): Target square size for resizing. Defaults to 224,
            which is the standard input size for most ViT models.

    Returns:
        PIL.Image: Preprocessed RGB image resized to (target_size, target_size).

    Example:
        >>> # From file path
        >>> img = preprocess_image("path/to/image.jpg")

        >>> # From PIL Image
        >>> from PIL import Image
        >>> img = Image.open("cat.jpg")
        >>> processed_img = preprocess_image(img, target_size=384)

    Note:
        - Grayscale and RGBA images are automatically converted to RGB
        - Maintains aspect ratio is not preserved; images are center-cropped and resized
        - No normalization is applied; use model processor for that
    """
    # If input is a file path string, load the image
    if isinstance(image, str):
        image = Image.open(image)

    # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image to target dimensions
    # Uses LANCZOS resampling for high-quality downsampling
    image = image.resize((target_size, target_size))

    return image


def normalize_heatmap(heatmap):
    """
    Normalize a heatmap array to the [0, 1] range using min-max scaling.

    This function is essential for visualizing heatmaps with consistent color mapping,
    regardless of the original value range. It handles edge cases where all values
    are identical.

    Args:
        heatmap (np.ndarray): Input heatmap array of any shape. Can contain any
            numeric values (int or float).

    Returns:
        np.ndarray: Normalized heatmap with values in [0, 1] range, preserving
            the original shape and relative differences between values.

    Example:
        >>> heatmap = np.array([[100, 200], [150, 250]])
        >>> normalized = normalize_heatmap(heatmap)
        >>> print(normalized)
        [[0.0, 0.666...], [0.333..., 1.0]]

        >>> # Edge case: all values are the same
        >>> constant = np.array([[5, 5], [5, 5]])
        >>> normalized = normalize_heatmap(constant)
        >>> print(normalized)
        [[0. 0.] [0. 0.]]

    Note:
        - Uses min-max normalization: (x - min) / (max - min)
        - Returns zeros if max equals min (constant heatmap)
        - Preserves NaN and inf values in the output
    """
    # Check if there's any variation in the heatmap
    if heatmap.max() > heatmap.min():
        # Apply min-max normalization to scale to [0, 1]
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        # If all values are the same, return zeros
        return np.zeros_like(heatmap)


def overlay_heatmap(image, heatmap, alpha=0.5, colormap="hot"):
    """
    Overlay a normalized heatmap on an original image with transparency blending.

    This function creates a visualization by blending a heatmap (e.g., attention map,
    saliency map) with the original image. The heatmap is colored using a matplotlib
    colormap and blended with the image using alpha transparency.

    Args:
        image (PIL.Image): Original RGB image to overlay the heatmap on.
        heatmap (np.ndarray): 2D array of heatmap values. Will be automatically
            normalized to [0, 1] range and resized to match image dimensions.
        alpha (float, optional): Transparency level for heatmap overlay.
            Range: [0, 1] where 0 = invisible, 1 = fully opaque. Defaults to 0.5.
        colormap (str, optional): Matplotlib colormap name for heatmap coloring.
            Common options: 'hot', 'jet', 'viridis', 'coolwarm'. Defaults to 'hot'.

    Returns:
        PIL.Image: RGB image with heatmap overlay, same size as input image.

    Example:
        >>> from PIL import Image
        >>> import numpy as np
        >>> image = Image.open("cat.jpg")
        >>> heatmap = np.random.rand(14, 14)  # Example attention map
        >>> overlay = overlay_heatmap(image, heatmap, alpha=0.6, colormap='jet')
        >>> overlay.save("cat_with_attention.jpg")

    Note:
        - Heatmap is automatically normalized to [0, 1] range
        - Heatmap is resized to match image dimensions using high-quality resampling
        - Supports any matplotlib colormap
        - Returns RGB image (alpha channel is removed after blending)
    """
    # Normalize heatmap to [0, 1] range for consistent coloring
    heatmap = normalize_heatmap(heatmap)

    # Convert heatmap to RGB using the specified matplotlib colormap
    # plt.cm.get_cmap() returns a colormap function
    cmap = plt.get_cmap(colormap)
    # Apply colormap and extract RGB channels (discard alpha)
    heatmap_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

    # Convert numpy array to PIL Image for resizing
    heatmap_img = Image.fromarray(heatmap_rgb)

    # Resize heatmap to match original image dimensions
    # Uses LANCZOS for high-quality upsampling/downsampling
    heatmap_img = heatmap_img.resize(image.size, Image.Resampling.LANCZOS)

    # Convert both images to RGBA for blending
    original_rgba = image.convert("RGBA")
    heatmap_rgba = heatmap_img.convert("RGBA")

    # Blend images using alpha transparency
    # alpha parameter controls the weight of heatmap vs original image
    blended = Image.blend(original_rgba, heatmap_rgba, alpha)

    # Convert back to RGB (remove alpha channel)
    return blended.convert("RGB")


def create_comparison_figure(original_image, explanation_images, explanation_titles):
    """
    Create a side-by-side comparison figure showing original image and multiple explanations.

    This function is useful for comparing different explainability methods (e.g., attention,
    GradCAM, SHAP) in a single visualization. All images are displayed with equal sizing
    and no axis ticks for a clean presentation.

    Args:
        original_image (PIL.Image): The original input image to display first.
        explanation_images (list): List of PIL Images containing explanation visualizations.
            Each should be the same size as the original image.
        explanation_titles (list): List of strings with titles for each explanation.
            Length must match explanation_images.

    Returns:
        matplotlib.figure.Figure: Figure object with (1 + n) subplots arranged horizontally,
            where n = len(explanation_images).

    Example:
        >>> original = Image.open("cat.jpg")
        >>> attention_map = generate_attention_viz(original)
        >>> gradcam_map = generate_gradcam_viz(original)
        >>>
        >>> fig = create_comparison_figure(
        ...     original,
        ...     [attention_map, gradcam_map],
        ...     ['Attention', 'GradCAM']
        ... )
        >>> fig.savefig('comparison.png')

    Note:
        - Automatically adjusts figure width based on number of images
        - All axes ticks are removed for cleaner visualization
        - Uses tight_layout() to prevent label overlap
    """
    # Calculate number of explanation images
    num_explanations = len(explanation_images)

    # Create figure with horizontal subplot layout
    # Width scales with number of images (4 inches per image)
    fig, axes = plt.subplots(
        1, num_explanations + 1, figsize=(4 * (num_explanations + 1), 4)  # +1 for original image
    )

    # Plot original image in first subplot
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")  # Remove axis ticks and labels

    # Plot each explanation image in subsequent subplots
    for i, (exp_img, title) in enumerate(zip(explanation_images, explanation_titles)):
        axes[i + 1].imshow(exp_img)
        axes[i + 1].set_title(title, fontweight="bold")
        axes[i + 1].axis("off")  # Remove axis ticks and labels

    # Adjust spacing to prevent title/label overlap
    plt.tight_layout()

    return fig


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a PIL Image.

    This utility function handles tensor-to-image conversion with automatic handling
    of batch dimensions, device placement (CPU/GPU), normalization, and channel ordering.
    Useful for visualizing model inputs, intermediate features, or generated images.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) or (B, C, H, W) where:
            - B = batch size (will be squeezed if present)
            - C = number of channels (typically 3 for RGB)
            - H = height in pixels
            - W = width in pixels

    Returns:
        PIL.Image: RGB image representation of the tensor.

    Example:
        >>> # Convert model input back to image
        >>> input_tensor = processor(image, return_tensors="pt")['pixel_values']
        >>> recovered_image = tensor_to_image(input_tensor)
        >>> recovered_image.show()

        >>> # Visualize intermediate feature map
        >>> feature_map = model.get_intermediate_features(input_tensor)
        >>> feature_img = tensor_to_image(feature_map)

    Note:
        - Automatically removes batch dimension if present (4D -> 3D)
        - Moves tensor to CPU if on GPU
        - Detaches tensor from computation graph
        - Normalizes values to [0, 1] range if outside this range
        - Converts from (C, H, W) to (H, W, C) format for PIL
        - Scales to [0, 255] and converts to uint8
    """
    # Remove batch dimension if present
    # Changes shape from (1, C, H, W) to (C, H, W)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Move tensor to CPU and detach from computation graph
    # This prevents gradient tracking and allows numpy conversion
    tensor = tensor.cpu().detach()

    # Normalize tensor to [0, 1] range if needed
    # Handles both normalized inputs (e.g., ImageNet normalization)
    # and unnormalized feature maps
    if tensor.min() < 0 or tensor.max() > 1:
        # Apply min-max normalization
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # Convert from PyTorch's (C, H, W) to numpy's (H, W, C) format
    numpy_image = tensor.permute(1, 2, 0).numpy()

    # Scale to [0, 255] range and convert to unsigned 8-bit integers
    numpy_image = (numpy_image * 255).astype(np.uint8)

    # Convert numpy array to PIL Image
    return Image.fromarray(numpy_image)


def get_top_predictions_dict(probs, labels, top_k=5):
    """
    Convert top predictions to a dictionary format for Gradio Label component.

    This convenience function formats prediction results for display in Gradio's
    Label component, which requires a dictionary mapping class names to probabilities.

    Args:
        probs (np.ndarray or list): Array or list of probability scores.
            Should be in descending order (highest probability first).
        labels (list): List of class names corresponding to probabilities.
            Must have same length as probs or longer.
        top_k (int, optional): Number of top predictions to include.
            Defaults to 5. If larger than length of probs/labels, uses maximum available.

    Returns:
        dict: Dictionary mapping class names (str) to probability scores (float).
            Keys are class labels, values are probabilities in range [0, 1].

    Example:
        >>> probs = np.array([0.87, 0.08, 0.03, 0.01, 0.01])
        >>> labels = ['tabby cat', 'tiger cat', 'Egyptian cat', 'lynx', 'cougar']
        >>> pred_dict = get_top_predictions_dict(probs, labels, top_k=3)
        >>> print(pred_dict)
        {'tabby cat': 0.87, 'tiger cat': 0.08, 'Egyptian cat': 0.03}

        >>> # Use with Gradio
        >>> import gradio as gr
        >>> output = gr.Label(label="Predictions")
        >>> # Can directly pass pred_dict to this component

    Note:
        - Probabilities are converted to Python float for JSON serialization
        - Only includes top_k predictions (useful for limiting display)
        - Maintains order from input (highest to lowest probability)
    """
    # Create dictionary by zipping labels with probabilities
    # Slicing [:top_k] limits to top_k predictions
    # float() conversion ensures JSON serialization compatibility
    return {label: float(prob) for label, prob in zip(labels[:top_k], probs[:top_k])}
