"""
Model Loader Module

This module handles loading Vision Transformer (ViT) models and their processors
from the Hugging Face model hub. It configures models for explainability by
enabling attention weight extraction.

Author: ViT-XAI-Dashboard Team
License: MIT
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor


def load_model_and_processor(model_name="google/vit-base-patch16-224"):
    """
    Load a Vision Transformer model and its corresponding image processor from Hugging Face.

    This function loads a pre-trained ViT model and configures it for explainability
    analysis by enabling attention weight outputs and using eager execution mode.
    The model is automatically moved to GPU if available.

    Args:
        model_name (str, optional): Hugging Face model identifier.
            Defaults to "google/vit-base-patch16-224".
            Examples:
                - "google/vit-base-patch16-224" (86M parameters)
                - "google/vit-large-patch16-224" (304M parameters)

    Returns:
        tuple: A tuple containing:
            - model (ViTForImageClassification): The loaded ViT model in eval mode
            - processor (ViTImageProcessor): The corresponding image processor

    Raises:
        Exception: If model loading fails due to network issues, invalid model name,
            or insufficient memory.

    Example:
        >>> model, processor = load_model_and_processor("google/vit-base-patch16-224")
        Loading model google/vit-base-patch16-224...
        ✅ Model and processor loaded successfully on cuda!

        >>> # Use with custom model
        >>> model, processor = load_model_and_processor("your-username/custom-vit")

    Note:
        - Model is automatically set to evaluation mode (no dropout, batch norm in eval)
        - Attention outputs are enabled for explainability methods
        - Uses "eager" attention implementation (not Flash Attention) to extract weights
        - GPU is used automatically if available, otherwise falls back to CPU
    """
    try:
        print(f"Loading model {model_name}...")

        # Load the image processor (handles image preprocessing and normalization)
        # This ensures images are correctly formatted for the model
        processor = ViTImageProcessor.from_pretrained(model_name)

        # Load the model with eager attention implementation
        # Note: "eager" mode is required to access attention weights for explainability
        # Flash Attention and other optimized implementations don't expose attention matrices
        model = ViTForImageClassification.from_pretrained(
            model_name, attn_implementation="eager"  # Enable attention weight extraction
        )

        # Enable attention output in model config
        # This makes attention weights available in forward pass outputs
        model.config.output_attentions = True

        # Determine device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Set model to evaluation mode
        # This disables dropout and sets batch normalization to eval mode
        model.eval()

        # Print success message with device info
        print(f"✅ Model and processor loaded successfully on {device}!")
        print(f"   Using attention implementation: {model.config._attn_implementation}")

        return model, processor

    except Exception as e:
        # Re-raise exception with context for debugging
        print(f"❌ Error loading model {model_name}: {str(e)}")
        raise


# Dictionary of supported ViT models with their Hugging Face identifiers
# Users can easily add more models by extending this dictionary
SUPPORTED_MODELS = {
    "ViT-Base": "google/vit-base-patch16-224",  # 86M params, good balance of speed/accuracy
    "ViT-Large": "google/vit-large-patch16-224",  # 304M params, higher accuracy but slower
}
