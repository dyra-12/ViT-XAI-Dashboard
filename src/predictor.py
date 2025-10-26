"""
Predictor Module

This module handles image classification predictions using Vision Transformer models.
It provides functions for making predictions and creating visualization plots of results.

Author: ViT-XAI-Dashboard Team
License: MIT
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def predict_image(image, model, processor, top_k=5):
    """
    Perform inference on an image and return top-k predicted classes with probabilities.

    This function takes a PIL Image, preprocesses it using the model's processor,
    performs a forward pass through the model, and returns the top-k most likely
    class predictions along with their confidence scores.

    Args:
        image (PIL.Image): Input image to classify. Should be in RGB format.
        model (ViTForImageClassification): Pre-trained ViT model for inference.
        processor (ViTImageProcessor): Image processor for preprocessing.
        top_k (int, optional): Number of top predictions to return. Defaults to 5.

    Returns:
        tuple: A tuple containing three elements:
            - top_probs (np.ndarray): Array of shape (top_k,) with confidence scores
            - top_indices (np.ndarray): Array of shape (top_k,) with class indices
            - top_labels (list): List of length top_k with human-readable class names

    Raises:
        Exception: If prediction fails due to invalid image, model issues, or memory errors.

    Example:
        >>> from PIL import Image
        >>> image = Image.open("cat.jpg")
        >>> probs, indices, labels = predict_image(image, model, processor, top_k=3)
        >>> print(f"Top prediction: {labels[0]} with {probs[0]:.2%} confidence")
        Top prediction: tabby cat with 87.34% confidence

    Note:
        - Inference is performed with torch.no_grad() for efficiency
        - Automatically handles device placement (CPU/GPU)
        - Applies softmax to convert logits to probabilities
    """
    try:
        # Get the device from the model parameters
        # This ensures inputs are moved to the same device as model (CPU or GPU)
        device = next(model.parameters()).device

        # Preprocess the image using the ViT processor
        # This handles resizing, normalization, and conversion to tensors
        inputs = processor(images=image, return_tensors="pt")

        # Move all input tensors to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform inference without gradient computation (saves memory and speeds up)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # Raw model outputs before softmax

        # Apply softmax to convert logits to probabilities
        # dim=-1 applies softmax across the class dimension
        probabilities = F.softmax(logits, dim=-1)[0]  # [0] removes batch dimension

        # Get the top-k highest probability predictions
        # Returns both values (probabilities) and indices (class IDs)
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # Convert PyTorch tensors to NumPy arrays for easier handling
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        # Convert class indices to human-readable labels using model's label mapping when available
        id2label = None
        if hasattr(model, "config") and hasattr(model.config, "id2label"):
            id2label = model.config.id2label

        top_labels = [
            (id2label.get(int(idx), f"class_{int(idx)}") if isinstance(id2label, dict) else f"class_{int(idx)}")
            for idx in top_indices
        ]

        return top_probs, top_indices, top_labels

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        raise


def create_prediction_plot(probs, labels):
    """
    Create a professional horizontal bar chart visualizing top predictions.

    This function generates a matplotlib figure with a horizontal bar chart showing
    the model's top predictions along with their confidence scores. The chart includes
    percentage labels on each bar and a clean, minimalist design.

    Args:
        probs (np.ndarray or list): Array of probability scores for each class.
            Should be in descending order (highest probability first).
        labels (list): List of human-readable class names corresponding to probabilities.
            Length must match probs.

    Returns:
        matplotlib.figure.Figure: A matplotlib Figure object containing the bar chart.
            Can be displayed with fig.show() or saved with fig.savefig().

    Example:
        >>> probs = np.array([0.87, 0.08, 0.03, 0.01, 0.01])
        >>> labels = ['tabby cat', 'tiger cat', 'Egyptian cat', 'lynx', 'cougar']
        >>> fig = create_prediction_plot(probs, labels)
        >>> fig.savefig('predictions.png')

    Note:
        - Uses horizontal bars for better label readability
        - Automatically adds percentage labels on each bar
        - Includes subtle grid lines for easier value reading
        - X-axis is scaled to provide padding for percentage labels
    """
    # Create figure and axis with specified size
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create horizontal bar chart
    # y_pos represents the vertical position of each bar
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, probs, color="skyblue", alpha=0.8)

    # Set y-axis ticks and labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)

    # Set axis labels and title
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_title("Top Predictions", fontsize=14, fontweight="bold")

    # Add probability percentage text on each bar
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()  # Get the bar length (probability value)
        # Place text slightly to the right of the bar end
        ax.text(
            width + 0.01,  # X position (slightly right of bar)
            bar.get_y() + bar.get_height() / 2,  # Y position (center of bar)
            f"{prob:.2%}",  # Format as percentage with 2 decimal places
            va="center",  # Vertical alignment
            fontsize=9,
        )

    # Set x-axis limits with padding for percentage labels
    # 1.15 multiplier adds 15% padding to the right
    ax.set_xlim(0, max(probs) * 1.15)

    # Add subtle grid lines for easier value reading
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig
