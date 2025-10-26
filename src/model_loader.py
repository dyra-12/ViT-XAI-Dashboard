"""
Model Loader Module

This module handles loading image classification models and their processors
from the Hugging Face model hub. It is optimized for ViT-style models but can
load a variety of architectures via Auto classes. For ViT models, it configures
the model for explainability by enabling attention weights.

Author: ViT-XAI-Dashboard Team
License: MIT
"""

import torch
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
)
from types import SimpleNamespace
import warnings


def load_model_and_processor(model_name="google/vit-base-patch16-224"):
    """
    Load an image classification model and its corresponding image processor from Hugging Face.

    This function uses the Transformers Auto classes to support multiple
    architectures (ViT, DeiT, Swin, ResNet, etc.). For ViT-like models, it
    enables attention weight outputs and prefers "eager" attention to make
    attention matrices accessible for explainability.

    Args:
        model_name (str, optional): Hugging Face model identifier.
            Defaults to "google/vit-base-patch16-224".

    Returns:
        tuple: (model, processor)
            - model (PreTrainedModel): The loaded model in eval mode
            - processor (ImageProcessor): The corresponding image processor

    Raises:
        Exception: If model loading fails due to network issues, invalid model name,
            or insufficient memory.

    Note:
        - Model is automatically set to evaluation mode
        - Attention outputs are enabled when the model supports them
        - For ViT-like models, we try to use the "eager" attention implementation
        - GPU is used automatically if available, otherwise falls back to CPU
    """
    try:
        print(f"Loading model {model_name}...")

        # Load the image processor (handles image preprocessing and normalization)
        processor = AutoImageProcessor.from_pretrained(model_name)

        # Load the model using Auto classes (supports many architectures)
        model = AutoModelForImageClassification.from_pretrained(model_name)

        # Enable attention output in model config when available
        # This makes attention weights available in forward pass outputs
        if hasattr(model, "config"):
            try:
                model.config.output_attentions = True
            except Exception:
                pass

            # Prefer "eager" attention implementation when the config supports it
            # This is particularly relevant for ViT models to expose attention weights
            for attr in ("_attn_implementation", "attn_implementation"):
                if hasattr(model.config, attr):
                    try:
                        setattr(model.config, attr, "eager")
                    except Exception:
                        pass

        # Determine device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Set model to evaluation mode
        # This disables dropout and sets batch normalization to eval mode
        model.eval()

        # Print success message with device info
        print(f"✅ Model and processor loaded successfully on {device}!")
        # Best-effort informational printout for attention implementation if available
        attn_impl = None
        if hasattr(model, "config"):
            for attr in ("_attn_implementation", "attn_implementation"):
                if hasattr(model.config, attr):
                    attn_impl = getattr(model.config, attr)
                    break
        if attn_impl is not None:
            print(f"   Using attention implementation: {attn_impl}")

        return model, processor

    except Exception as e:
        # Handle known EfficientNet issue that requires torch>=2.6 for torch.load
        err_msg = str(e)
        print(f"⚠️ Primary load failed for {model_name}: {err_msg}")

        if "efficientnet" in model_name.lower() or "v2.6" in err_msg:
            try:
                print("Attempting fallback to timm for EfficientNet...")
                model, processor = _load_efficientnet_with_timm(model_name)
                # Move to device and eval as usual
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                model.eval()
                print(f"✅ Fallback loaded via timm on {device}!")
                return model, processor
            except Exception as ee:
                print(f"❌ Fallback via timm failed: {ee}")
                raise

        # Re-raise exception with context for debugging if not handled
        print(f"❌ Error loading model {model_name}: {str(e)}")
        raise


class _SimpleImageProcessor:
    """Minimal image processor to mimic HF processor for non-HF models.

    Returns a dict with 'pixel_values' suitable for our predictor pipeline.
    """

    def __init__(self, size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        from torchvision import transforms

        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, images, return_tensors="pt"):
        if return_tensors != "pt":
            warnings.warn("_SimpleImageProcessor only supports return_tensors='pt'")
        import torch as _torch
        # Expect a single PIL Image for our use-cases
        tensor = self.transform(images).unsqueeze(0)  # (1, C, H, W)
        return {"pixel_values": tensor}


class _HFLikeOutput:
    def __init__(self, logits):
        self.logits = logits


class _HFLikeModelWrapper(torch.nn.Module):
    """Wrap a timm model to present an HF-like interface with config.id2label.

    Forward accepts pixel_values and returns an object with .logits
    """

    def __init__(self, model, id2label):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(id2label=id2label)

    def forward(self, pixel_values):
        logits = self.model(pixel_values)
        return _HFLikeOutput(logits)


def _load_efficientnet_with_timm(model_name: str):
    """Load EfficientNet via timm as a fallback, returning (model, processor)."""
    try:
        import timm
    except Exception as e:
        raise RuntimeError(
            "timm is required for EfficientNet fallback. Please install 'timm'."
        ) from e

    # Map HF name to a commonly available timm variant
    variant = "tf_efficientnet_b7_ns" if "b7" in model_name.lower() else "tf_efficientnet_b0"
    net = timm.create_model(variant, pretrained=True, num_classes=1000)
    net.eval()

    # Build ImageNet-1k id2label mapping if needed
    id2label = {i: f"class_{i}" for i in range(1000)}

    wrapped = _HFLikeModelWrapper(net, id2label)
    processor = _SimpleImageProcessor(size=224)
    return wrapped, processor


# Dictionary of supported ViT models with their Hugging Face identifiers
# Users can easily add more models by extending this dictionary
SUPPORTED_MODELS = {
    # ViT family
    "ViT-Base": "google/vit-base-patch16-224",  # 86M params, good balance of speed/accuracy
    "ViT-Large": "google/vit-large-patch16-224",  # 304M params, higher accuracy but slower

    # New additions
    "ResNet-50": "microsoft/resnet-50",
    "Swin Transformer": "microsoft/swin-base-patch4-window7-224",
    "DeiT": "facebook/deit-base-patch16-224",
    "EfficientNet": "google/efficientnet-b7",  # Note: may have limited attention-based XAI
}
