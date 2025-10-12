# app.py

import gradio as gr
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_loader import load_model_and_processor, SUPPORTED_MODELS
from predictor import predict_image, create_prediction_plot
from explainer import explain_attention, explain_gradcam, explain_gradient_shap
from auditor import create_auditors
from utils import preprocess_image, get_top_predictions_dict

# Global variables to cache model and processor
model = None
processor = None
current_model_name = None
auditors = None

def load_selected_model(model_name):
    """Load the selected model and cache it globally."""
    global model, processor, current_model_name, auditors
    
    try:
        if model is None or current_model_name != model_name:
            print(f"Loading model: {model_name}")
            model, processor = load_model_and_processor(model_name)
            current_model_name = model_name
            
            # Initialize auditors
            auditors = create_auditors(model, processor)
            print("✅ Model and auditors loaded successfully!")
        
        return f"✅ Model loaded: {model_name}"
    
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def analyze_image_basic(image, model_choice, xai_method, layer_index, head_index):
    """
    Basic explainability analysis - the core function for Tab 1.
    """
    try:
        # Load model if needed
        model_status = load_selected_model(SUPPORTED_MODELS[model_choice])
        if "❌" in model_status:
            return None, None, None, model_status
        
        # Preprocess image
        if image is None:
            return None, None, None, "⚠️ Please upload an image first."
        
        processed_image = preprocess_image(image)
        
        # Get predictions
        probs, indices, labels = predict_image(processed_image, model, processor)
        pred_fig = create_prediction_plot(probs, labels)
        
        # Generate explanation based on selected method
        explanation_fig = None
        explanation_image = None
        
        if xai_method == "Attention Visualization":
            explanation_fig = explain_attention(
                model, processor, processed_image, 
                layer_index=layer_index, head_index=head_index
            )
            
        elif xai_method == "GradCAM":
            explanation_fig, explanation_image = explain_gradcam(
                model, processor, processed_image
            )
            
        elif xai_method == "GradientSHAP":
            explanation_fig = explain_gradient_shap(
                model, processor, processed_image, n_samples=3
            )
        
        # Convert predictions to dictionary for Gradio Label
        pred_dict = get_top_predictions_dict(probs, labels)
        
        return processed_image, pred_fig, explanation_fig, f"✅ Analysis complete! Top prediction: {labels[0]} ({probs[0]:.2%})"
    
    except Exception as e:
        error_msg = f"❌ Analysis failed: {str(e)}"
        print(error_msg)
        return None, None, None, error_msg

def analyze_counterfactual(image, model_choice, patch_size, perturbation_type):
    """
    Counterfactual analysis for Tab 2.
    """
    try:
        # Load model if needed
        model_status = load_selected_model(SUPPORTED_MODELS[model_choice])
        if "❌" in model_status:
            return None, None, model_status
        
        if image is None:
            return None, None, "⚠️ Please upload an image first."
        
        processed_image = preprocess_image(image)
        
        # Perform counterfactual analysis
        results = auditors['counterfactual'].patch_perturbation_analysis(
            processed_image, 
            patch_size=patch_size,
            perturbation_type=perturbation_type
        )
        
        # Create summary message
        summary = (
            f"🔍 Counterfactual Analysis Complete!\n"
            f"• Avg confidence change: {results['avg_confidence_change']:.4f}\n"
            f"• Prediction flip rate: {results['prediction_flip_rate']:.2%}\n"
            f"• Most sensitive patch: {results['most_sensitive_patch']}"
        )
        
        return results['figure'], summary
    
    except Exception as e:
        error_msg = f"❌ Counterfactual analysis failed: {str(e)}"
        print(error_msg)
        return None, error_msg

def analyze_calibration(image, model_choice, n_bins):
    """
    Confidence calibration analysis for Tab 3.
    """
    try:
        # Load model if needed
        model_status = load_selected_model(SUPPORTED_MODELS[model_choice])
        if "❌" in model_status:
            return None, None, model_status
        
        if image is None:
            return None, None, "⚠️ Please upload an image first."
        
        processed_image = preprocess_image(image)
        
        # For demo purposes, create a simple test set from the uploaded image
        # In a real scenario, you'd use a proper validation set
        test_images = [processed_image] * 10  # Create multiple copies
        
        # Perform calibration analysis
        results = auditors['calibration'].analyze_calibration(
            test_images, n_bins=n_bins
        )
        
        # Create summary message
        metrics = results['metrics']
        summary = (
            f"📊 Calibration Analysis Complete!\n"
            f"• Mean confidence: {metrics['mean_confidence']:.3f}\n"
            f"• Overconfident rate: {metrics['overconfident_rate']:.2%}\n"
            f"• Underconfident rate: {metrics['underconfident_rate']:.2%}"
        )
        
        return results['figure'], summary
    
    except Exception as e:
        error_msg = f"❌ Calibration analysis failed: {str(e)}"
        print(error_msg)
        return None, error_msg

def analyze_bias_detection(image, model_choice):
    """
    Bias detection analysis for Tab 4.
    """
    try:
        # Load model if needed
        model_status = load_selected_model(SUPPORTED_MODELS[model_choice])
        if "❌" in model_status:
            return None, None, model_status
        
        if image is None:
            return None, None, "⚠️ Please upload an image first."
        
        processed_image = preprocess_image(image)
        
        # Create demo subgroups based on the uploaded image
        # In a real scenario, you'd use predefined subgroups from your dataset
        subsets = []
        subset_names = ['Original', 'Brightness+', 'Brightness-', 'Contrast+']
        
        # Original image
        subsets.append([processed_image])
        
        # Brightness increased
        bright_image = processed_image.copy().point(lambda p: min(255, p * 1.5))
        subsets.append([bright_image])
        
        # Brightness decreased
        dark_image = processed_image.copy().point(lambda p: p * 0.7)
        subsets.append([dark_image])
        
        # Contrast increased
        contrast_image = processed_image.copy().point(lambda p: 128 + (p - 128) * 1.5)
        subsets.append([contrast_image])
        
        # Perform bias analysis
        results = auditors['bias'].analyze_subgroup_performance(
            subsets, subset_names
        )
        
        # Create summary message
        subgroup_metrics = results['subgroup_metrics']
        summary = f"⚖️ Bias Detection Complete!\nAnalyzed {len(subgroup_metrics)} subgroups:\n"
        
        for name, metrics in subgroup_metrics.items():
            summary += f"• {name}: confidence={metrics['mean_confidence']:.3f}\n"
        
        return results['figure'], summary
    
    except Exception as e:
        error_msg = f"❌ Bias detection failed: {str(e)}"
        print(error_msg)
        return None, error_msg

def create_demo_image():
    """Create a demo image for first-time users."""
    # Create a simple demo image with multiple colors
    img = Image.new('RGB', (224, 224), color=(150, 100, 100))
    
    # Add different colored regions
    for x in range(50, 150):
        for y in range(50, 150):
            img.putpixel((x, y), (100, 200, 100))  # Green square
    
    for x in range(160, 200):
        for y in range(160, 200):
            img.putpixel((x, y), (100, 100, 200))  # Blue square
    
    return img

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="ViT Auditing Toolkit") as demo:
    gr.Markdown(
        """
        # 🎯 ViT Auditing Toolkit
        ### An Interactive Dashboard for Model Explainability and Validation
        
        Upload an image or use the demo image to analyze Vision Transformer model predictions 
        and explore various explanation methods.
        """
    )
    
    # Model selection (shared across all tabs)
    with gr.Row():
        model_choice = gr.Dropdown(
            choices=list(SUPPORTED_MODELS.keys()),
            value="ViT-Base",
            label="🎯 Select Model",
            info="Choose which Vision Transformer model to use"
        )
        
        load_btn = gr.Button("🔄 Load Model", variant="primary")
        model_status = gr.Textbox(label="Model Status", interactive=False)
    
    load_btn.click(
        fn=lambda model: load_selected_model(SUPPORTED_MODELS[model]),
        inputs=[model_choice],
        outputs=[model_status]
    )
    
    # Tabbed interface
    with gr.Tabs():
        # Tab 1: Basic Explainability
        with gr.TabItem("🔍 Basic Explainability"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📁 Upload Image",
                        type="pil",
                        value=create_demo_image()
                    )
                    
                    with gr.Accordion("⚙️ Explanation Settings", open=False):
                        xai_method = gr.Dropdown(
                            choices=[
                                "Attention Visualization", 
                                "GradCAM", 
                                "GradientSHAP"
                            ],
                            value="Attention Visualization",
                            label="Explanation Method"
                        )
                        
                        with gr.Row():
                            layer_index = gr.Slider(
                                minimum=0, maximum=11, value=6, step=1,
                                label="Attention Layer Index"
                            )
                            head_index = gr.Slider(
                                minimum=0, maximum=11, value=0, step=1,
                                label="Attention Head Index"
                            )
                    
                    analyze_btn = gr.Button("🚀 Analyze Image", variant="primary")
                    status_output = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=2):
                    with gr.Row():
                        original_display = gr.Image(
                            label="📸 Processed Image",
                            interactive=False
                        )
                        prediction_display = gr.Plot(
                            label="📊 Model Predictions"
                        )
                    
                    explanation_display = gr.Plot(
                        label="🔍 Explanation Visualization"
                    )
            
            # Connect the analyze button
            analyze_btn.click(
                fn=analyze_image_basic,
                inputs=[image_input, model_choice, xai_method, layer_index, head_index],
                outputs=[original_display, prediction_display, explanation_display, status_output]
            )
        
        # Tab 2: Counterfactual Analysis
        with gr.TabItem("🔄 Counterfactual Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    cf_image_input = gr.Image(
                        label="📁 Upload Image",
                        type="pil",
                        value=create_demo_image()
                    )
                    
                    with gr.Accordion("⚙️ Counterfactual Settings", open=True):
                        patch_size = gr.Slider(
                            minimum=16, maximum=64, value=32, step=16,
                            label="Patch Size"
                        )
                        perturbation_type = gr.Dropdown(
                            choices=["blur", "blackout", "gray", "noise"],
                            value="blur",
                            label="Perturbation Type"
                        )
                    
                    cf_analyze_btn = gr.Button("🔄 Run Counterfactual Analysis", variant="primary")
                    cf_status_output = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=2):
                    cf_explanation_display = gr.Plot(
                        label="🔄 Counterfactual Analysis Results"
                    )
            
            cf_analyze_btn.click(
                fn=analyze_counterfactual,
                inputs=[cf_image_input, model_choice, patch_size, perturbation_type],
                outputs=[cf_explanation_display, cf_status_output]
            )
        
        # Tab 3: Confidence Calibration
        with gr.TabItem("📊 Confidence Calibration"):
            with gr.Row():
                with gr.Column(scale=1):
                    cal_image_input = gr.Image(
                        label="📁 Upload Sample Image (Used to generate demo test set)",
                        type="pil",
                        value=create_demo_image()
                    )
                    
                    with gr.Accordion("⚙️ Calibration Settings", open=True):
                        n_bins = gr.Slider(
                            minimum=5, maximum=20, value=10, step=1,
                            label="Number of Bins"
                        )
                    
                    cal_analyze_btn = gr.Button("📊 Analyze Calibration", variant="primary")
                    cal_status_output = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=2):
                    cal_explanation_display = gr.Plot(
                        label="📊 Calibration Analysis Results"
                    )
            
            cal_analyze_btn.click(
                fn=analyze_calibration,
                inputs=[cal_image_input, model_choice, n_bins],
                outputs=[cal_explanation_display, cal_status_output]
            )
        
        # Tab 4: Bias Detection
        with gr.TabItem("⚖️ Bias Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    bias_image_input = gr.Image(
                        label="📁 Upload Sample Image (Used to generate demo subgroups)",
                        type="pil",
                        value=create_demo_image()
                    )
                    
                    bias_analyze_btn = gr.Button("⚖️ Detect Bias", variant="primary")
                    bias_status_output = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=2):
                    bias_explanation_display = gr.Plot(
                        label="⚖️ Bias Detection Results"
                    )
            
            bias_analyze_btn.click(
                fn=analyze_bias_detection,
                inputs=[bias_image_input, model_choice],
                outputs=[bias_explanation_display, bias_status_output]
            )
    
    # Footer
    gr.Markdown(
        """
        ---
        ### 🛠️ About This Toolkit
        
        This interactive dashboard provides comprehensive auditing capabilities for Vision Transformer models:
        
        - **🔍 Basic Explainability**: Understand model predictions with attention maps, GradCAM, and SHAP
        - **🔄 Counterfactual Analysis**: Test how predictions change with image perturbations  
        - **📊 Confidence Calibration**: Evaluate if the model is properly calibrated
        - **⚖️ Bias Detection**: Identify performance variations across different subgroups
        
        Built with ❤️ using Gradio, Transformers, and Captum.
        """
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(
        server_name="localhost",  # Changed from "0.0.0.0"
        server_port=7860,
        share=False,
        show_error=True
    )