# src/model_loader.py

from transformers import ViTImageProcessor, ViTForImageClassification
import torch

def load_model_and_processor(model_name="google/vit-base-patch16-224"):
    """
    Load a Vision Transformer model and its corresponding processor from Hugging Face.
    """
    try:
        print(f"Loading model {model_name}...")
        
        # Load processor and model with eager attention implementation
        processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Force eager attention implementation to get attention weights
        model = ViTForImageClassification.from_pretrained(
            model_name,
            attn_implementation="eager"  # This enables attention output
        )
        
        # Now we can safely set output_attentions
        model.config.output_attentions = True
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        print(f"✅ Model and processor loaded successfully on {device}!")
        print(f"   Using attention implementation: {model.config._attn_implementation}")
        return model, processor
        
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise

# Supported models
SUPPORTED_MODELS = {
    "ViT-Base": "google/vit-base-patch16-224",
    "ViT-Large": "google/vit-large-patch16-224",
}