# src/explainer.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import captum
from captum.attr import LayerGradCam, GradientShap
from captum.attr import visualization as viz
import torch.nn.functional as F

class ViTWrapper(torch.nn.Module):
    """
    Wrapper class to make Hugging Face ViT compatible with Captum.
    This returns raw tensors instead of Hugging Face output objects.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Hugging Face models expect pixel_values key
        outputs = self.model(pixel_values=x)
        return outputs.logits

class AttentionHook:
    """Hook to capture attention weights from ViT model"""
    def __init__(self):
        self.attention_weights = None
        
    def __call__(self, module, input, output):
        # For ViT, attention weights are usually the second output
        if len(output) >= 2:
            self.attention_weights = output[1]  # attention weights
        else:
            self.attention_weights = None

def explain_attention(model, processor, image, layer_index=6, head_index=0):
    """
    Extract and visualize attention weights using hooks.
    """
    try:
        device = next(model.parameters()).device
        
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Register hook to capture attention
        hook = AttentionHook()
        
        # Try different layer access patterns
        try:
            # For standard ViT structure
            target_layer = model.vit.encoder.layer[layer_index].attention.attention
            handle = target_layer.register_forward_hook(hook)
        except:
            try:
                # Alternative structure
                target_layer = model.vit.encoder.layers[layer_index].attention.attention
                handle = target_layer.register_forward_hook(hook)
            except:
                raise ValueError(f"Could not access layer {layer_index} for attention hook")
        
        # Forward pass to capture attention
        with torch.no_grad():
            _ = model(**inputs)
        
        # Remove hook
        handle.remove()
        
        if hook.attention_weights is None:
            raise ValueError("No attention weights captured by hook")
        
        # Get attention weights
        attention_weights = hook.attention_weights  # Shape: (batch, heads, seq_len, seq_len)
        attention_map = attention_weights[0, head_index]  # Shape: (seq_len, seq_len)
        
        # Remove CLS token attention to other tokens
        patch_attention = attention_map[1:, 1:]  # Remove CLS token rows and columns
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Display attention matrix
        im = ax.imshow(patch_attention.cpu().numpy(), cmap='viridis', aspect='auto')
        
        ax.set_title(f'Attention Map - Layer {layer_index}, Head {head_index}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Patches')
        ax.set_ylabel('Query Patches')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in attention visualization: {str(e)}")
        # Return a simple error plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Attention visualization failed:\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Attention Visualization Error')
        return fig

def explain_gradcam(model, processor, image, target_layer_index=-2):
    """
    Generate GradCAM heatmap for the predicted class.
    """
    try:
        device = next(model.parameters()).device
        
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")
        input_tensor = inputs['pixel_values'].to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_class = outputs.logits.argmax(dim=1).item()
        
        # Get the target layer
        try:
            target_layer = model.vit.encoder.layer[target_layer_index].attention.attention
        except:
            target_layer = model.vit.encoder.layers[target_layer_index].attention.attention
        
        # Create wrapped model for Captum compatibility
        wrapped_model = ViTWrapper(model)
        
        # Initialize GradCAM with wrapped model
        gradcam = LayerGradCam(wrapped_model, target_layer)
        
        # Generate attribution - handle tuple output
        attribution = gradcam.attribute(input_tensor, target=predicted_class)
        
        # FIX: Handle tuple output by taking the first element
        if isinstance(attribution, tuple):
            attribution = attribution[0]
        
        # Convert attribution to heatmap
        attribution = attribution.squeeze().cpu().detach().numpy()
        
        # Normalize attribution
        if attribution.max() > attribution.min():
            attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
        else:
            attribution = np.zeros_like(attribution)
        
        # Resize heatmap to match original image
        original_size = image.size
        heatmap = Image.fromarray((attribution * 255).astype(np.uint8))
        heatmap = heatmap.resize(original_size, Image.Resampling.LANCZOS)
        heatmap = np.array(heatmap)
        
        # Create visualization figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Heatmap
        ax2.imshow(heatmap, cmap='hot')
        ax2.set_title('GradCAM Heatmap')
        ax2.axis('off')
        
        # Overlay
        ax3.imshow(image)
        ax3.imshow(heatmap, cmap='hot', alpha=0.5)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        
        # Create overlay image for dashboard
        heatmap_rgb = (plt.cm.hot(heatmap / 255.0)[:, :, :3] * 255).astype(np.uint8)
        overlay_img = Image.fromarray(heatmap_rgb)
        overlay_img = overlay_img.resize(original_size, Image.Resampling.LANCZOS)
        
        # Blend with original
        original_rgba = image.convert('RGBA')
        overlay_rgba = overlay_img.convert('RGBA')
        blended = Image.blend(original_rgba, overlay_rgba, alpha=0.5)
        
        return fig, blended.convert('RGB')
        
    except Exception as e:
        print(f"Error in GradCAM: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"GradCAM failed:\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('GradCAM Error')
        return fig, image

def explain_gradient_shap(model, processor, image, n_samples=5):
    """
    Generate GradientSHAP explanations.
    """
    try:
        device = next(model.parameters()).device
        
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")
        input_tensor = inputs['pixel_values'].to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_class = outputs.logits.argmax(dim=1).item()
        
        # Create baseline (black image)
        baseline = torch.zeros_like(input_tensor)
        
        # Create wrapped model for Captum compatibility
        wrapped_model = ViTWrapper(model)
        
        # Initialize GradientSHAP with wrapped model
        gradient_shap = GradientShap(wrapped_model)
        
        # Generate attribution
        attribution = gradient_shap.attribute(
            input_tensor,
            baselines=baseline,
            n_samples=n_samples,
            target=predicted_class
        )
        
        # Summarize attribution across channels
        attribution = attribution.squeeze().sum(dim=0).cpu().detach().numpy()
        
        # Normalize
        if attribution.max() > attribution.min():
            attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
        else:
            attribution = np.zeros_like(attribution)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # SHAP attribution
        im = ax2.imshow(attribution, cmap='coolwarm')
        ax2.set_title('GradientSHAP Attribution')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        # Overlay
        ax3.imshow(image, alpha=0.7)
        im_overlay = ax3.imshow(attribution, cmap='coolwarm', alpha=0.5)
        ax3.set_title('Attribution Overlay')
        ax3.axis('off')
        plt.colorbar(im_overlay, ax=ax3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in GradientSHAP: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"GradientSHAP failed:\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('GradientSHAP Error')
        return fig