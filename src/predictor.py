# src/predictor.py

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def predict_image(image, model, processor, top_k=5):
    """
    Perform inference on an image and return top-k predictions.
    
    Args:
        image (PIL.Image): Input image to classify.
        model: Loaded ViT model.
        processor: Loaded ViT processor.
        top_k (int): Number of top predictions to return.
    
    Returns:
        tuple: (top_probs, top_indices, top_labels) - Probabilities, class indices, and label names.
    """
    try:
        # Get the device from the model
        device = next(model.parameters()).device
        
        # Preprocess the image - note: current processors return pixel_values
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=-1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Convert to Python lists and numpy arrays
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        # Get human-readable labels
        top_labels = [model.config.id2label[idx] for idx in top_indices]
        
        return top_probs, top_indices, top_labels
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def create_prediction_plot(probs, labels):
    """
    Create a clean, professional bar chart for top predictions.
    
    Args:
        probs (np.array): Array of probabilities.
        labels (list): List of label names.
    
    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, probs, color='skyblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_title('Top Predictions', fontsize=14, fontweight='bold')
    
    # Add probability text on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.2%}', va='center', fontsize=9)
    
    # Set x-axis limit and style
    ax.set_xlim(0, max(probs) * 1.15)  # Add some padding for text
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig