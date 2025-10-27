# app.py

import os
import sys
import time

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from auditor import create_auditors
from explainer import explain_attention, explain_gradcam, explain_gradient_shap
from model_loader import SUPPORTED_MODELS, load_model_and_processor
from predictor import create_prediction_plot, predict_image
from utils import get_top_predictions_dict, preprocess_image

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
                model, processor, processed_image, layer_index=layer_index, head_index=head_index
            )

        elif xai_method == "GradCAM":
            explanation_fig, explanation_image = explain_gradcam(model, processor, processed_image)

        elif xai_method == "GradientSHAP":
            explanation_fig = explain_gradient_shap(model, processor, processed_image, n_samples=3)

        # Convert predictions to dictionary for Gradio Label
        pred_dict = get_top_predictions_dict(probs, labels)

        return (
            processed_image,
            pred_fig,
            explanation_fig,
            f"✅ Analysis complete! Top prediction: {labels[0]} ({probs[0]:.2%})",
        )

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
        results = auditors["counterfactual"].patch_perturbation_analysis(
            processed_image, patch_size=patch_size, perturbation_type=perturbation_type
        )

        # Create summary message
        summary = (
            f"🔍 Counterfactual Analysis Complete!\n"
            f"• Avg confidence change: {results['avg_confidence_change']:.4f}\n"
            f"• Prediction flip rate: {results['prediction_flip_rate']:.2%}\n"
            f"• Most sensitive patch: {results['most_sensitive_patch']}"
        )

        return results["figure"], summary

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
        results = auditors["calibration"].analyze_calibration(test_images, n_bins=n_bins)

        # Create summary message
        metrics = results["metrics"]
        summary = (
            f"📊 Calibration Analysis Complete!\n"
            f"• Mean confidence: {metrics['mean_confidence']:.3f}\n"
            f"• Overconfident rate: {metrics['overconfident_rate']:.2%}\n"
            f"• Underconfident rate: {metrics['underconfident_rate']:.2%}"
        )

        return results["figure"], summary

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
        subset_names = ["Original", "Brightness+", "Brightness-", "Contrast+"]

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
        results = auditors["bias"].analyze_subgroup_performance(subsets, subset_names)

        # Create summary message
        subgroup_metrics = results["subgroup_metrics"]
        summary = f"⚖️ Bias Detection Complete!\nAnalyzed {len(subgroup_metrics)} subgroups:\n"

        for name, metrics in subgroup_metrics.items():
            summary += f"• {name}: confidence={metrics['mean_confidence']:.3f}\n"

        return results["figure"], summary

    except Exception as e:
        error_msg = f"❌ Bias detection failed: {str(e)}"
        print(error_msg)
        return None, error_msg


def create_demo_image():
    """Create a demo image for first-time users."""
    # Create a simple demo image with multiple colors
    img = Image.new("RGB", (224, 224), color=(150, 100, 100))

    # Add different colored regions
    for x in range(50, 150):
        for y in range(50, 150):
            img.putpixel((x, y), (100, 200, 100))  # Green square

    for x in range(160, 200):
        for y in range(160, 200):
            img.putpixel((x, y), (100, 100, 200))  # Blue square

    return img


# Minimal CSS for basic styling without breaking functionality
custom_css = """
/* Basic styling without interfering with dropdowns */
.gradio-container {
    background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
    font-family: 'Inter', sans-serif;
}

/* Header styling */
.main-header {
    background: rgba(99, 102, 241, 0.05);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 2rem;
}

/* Button styling */
button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none;
    color: white;
    font-weight: 600;
    padding: 14px 32px;
    border-radius: 12px;
}

button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.6);
}

/* Block styling */
.block {
    background: rgba(30, 41, 59, 0.4);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(99, 102, 241, 0.15);
}

/* Tab styling */
.tab-nav button {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 14px 28px;
    margin: 0 6px;
    color: #94a3b8;
    font-weight: 600;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
}
"""

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="ViT Auditing Toolkit") as demo:
    # Main Header
    gr.HTML(
        """
        <div class="main-header">
            <h1 style="
                font-size: 3rem; 
                font-weight: 800; 
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
                text-align: center;
            ">
                🎯 ViT Auditing Toolkit
            </h1>
            <p style="
                font-size: 1.25rem; 
                color: #94a3b8; 
                text-align: center;
                font-weight: 500;
                margin-bottom: 0;
            ">
                Comprehensive Model Explainability and Validation Dashboard
            </p>
        </div>
        """
    )

    # About Section
    gr.HTML(
        """
        <div style="
            background: rgba(30, 41, 59, 0.4);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(99, 102, 241, 0.15);
        ">
            <h2 style="font-size: 1.75rem; font-weight: 700; color: #e0e7ff; margin-bottom: 1rem;">
                ℹ️ About This Toolkit
            </h2>
            
            <p style="color: #94a3b8; line-height: 1.8; font-size: 1.05rem; margin-bottom: 1.5rem;">
                This interactive dashboard provides comprehensive auditing capabilities for Vision Transformer models, 
                enabling researchers and practitioners to understand, validate, and improve their AI models through 
                multiple explainability techniques.
            </p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem;">
                <div style="background: rgba(99, 102, 241, 0.08); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                    <strong style="color: #a5b4fc; font-size: 1.1rem;">Basic Explainability</strong>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;">
                        Understand model predictions with attention maps, GradCAM, and SHAP visualizations
                    </p>
                </div>
                
                <div style="background: rgba(99, 102, 241, 0.08); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔄</div>
                    <strong style="color: #c4b5fd; font-size: 1.1rem;">Counterfactual Analysis</strong>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;">
                        Test prediction robustness by systematically perturbing image regions
                    </p>
                </div>
                
                <div style="background: rgba(99, 102, 241, 0.08); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                    <strong style="color: #f9a8d4; font-size: 1.1rem;">Confidence Calibration</strong>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;">
                        Evaluate whether model confidence scores accurately reflect prediction reliability
                    </p>
                </div>
                
                <div style="background: rgba(99, 102, 241, 0.08); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚖️</div>
                    <strong style="color: #93c5fd; font-size: 1.1rem;">Bias Detection</strong>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;">
                        Identify performance variations across different demographic or data subgroups
                    </p>
                </div>
            </div>
        </div>
        """
    )

    # Quick Start Guide
    gr.HTML(
        """
        <div style="
            background: rgba(99, 102, 241, 0.1);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(99, 102, 241, 0.25);
        ">
            <h2 style="font-size: 1.5rem; font-weight: 700; color: #e0e7ff; margin-bottom: 1.5rem;">
                🚀 Quick Start Guide
            </h2>
            
            <div style="display: grid; gap: 1rem;">
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="
                        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                        border-radius: 50%;
                        width: 32px;
                        height: 32px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 700;
                        color: white;
                        flex-shrink: 0;
                    ">1</div>
                    <div>
                        <strong style="color: #c4b5fd; font-size: 1.05rem;">Select a Model</strong>
                        <p style="color: #94a3b8; margin-top: 0.25rem; line-height: 1.6;">
                            Choose a Vision Transformer model from the dropdown and click "Load Model" button
                        </p>
                    </div>
                </div>
                
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="
                        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                        border-radius: 50%;
                        width: 32px;
                        height: 32px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 700;
                        color: white;
                        flex-shrink: 0;
                    ">2</div>
                    <div>
                        <strong style="color: #c4b5fd; font-size: 1.05rem;">Upload Your Image</strong>
                        <p style="color: #94a3b8; margin-top: 0.25rem; line-height: 1.6;">
                            Navigate to any tab and upload an image you want to analyze
                        </p>
                    </div>
                </div>
                
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="
                        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                        border-radius: 50%;
                        width: 32px;
                        height: 32px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 700;
                        color: white;
                        flex-shrink: 0;
                    ">3</div>
                    <div>
                        <strong style="color: #c4b5fd; font-size: 1.05rem;">Choose Analysis Type</strong>
                        <p style="color: #94a3b8; margin-top: 0.25rem; line-height: 1.6;">
                            Select from 4 tabs: Basic Explainability, Counterfactual Analysis, Confidence Calibration, or Bias Detection
                        </p>
                    </div>
                </div>
                
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="
                        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                        border-radius: 50%;
                        width: 32px;
                        height: 32px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 700;
                        color: white;
                        flex-shrink: 0;
                    ">4</div>
                    <div>
                        <strong style="color: #c4b5fd; font-size: 1.05rem;">Run Analysis</strong>
                        <p style="color: #94a3b8; margin-top: 0.25rem; line-height: 1.6;">
                            Adjust settings if needed, then click the analysis button to see results and visualizations
                        </p>
                    </div>
                </div>
            </div>
            
            <div style="
                margin-top: 1.5rem;
                padding: 1rem;
                background: rgba(139, 92, 246, 0.1);
                border-radius: 12px;
                border-left: 4px solid #8b5cf6;
            ">
                <p style="color: #c4b5fd; margin: 0; font-size: 0.95rem;">
                    💡 <strong>Tip:</strong> Start with "Basic Explainability" to understand what your model sees, 
                    then explore advanced auditing features in other tabs.
                </p>
            </div>
        </div>
        """
    )

    # Model selection (shared across all tabs)
    with gr.Row():
        with gr.Column(scale=3):
            model_choice = gr.Dropdown(
                choices=list(SUPPORTED_MODELS.keys()),
                value="ViT-Base",
                label="🎯 Select Model",
                info="Choose which Vision Transformer model to use",
            )

        with gr.Column(scale=3):
            model_status = gr.Textbox(
                label="📡 Model Status",
                interactive=False,
                placeholder="Select a model and click 'Load Model' to begin...",
            )

        with gr.Column(scale=2):
            load_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")

    load_btn.click(
        fn=lambda model: load_selected_model(SUPPORTED_MODELS[model]),
        inputs=[model_choice],
        outputs=[model_status],
    )

    # Tabbed interface
    with gr.Tabs():
        # Tab 1: Basic Explainability
        with gr.TabItem("🔍 Basic Explainability"):
            gr.Markdown(
                """
                ### Understanding Model Predictions
                Visualize what the model "sees" and understand which features influence its decisions.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📁 Upload Image",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=350,
                    )

                    with gr.Accordion("⚙️ Explanation Settings", open=False):
                        xai_method = gr.Dropdown(
                            choices=["Attention Visualization", "GradCAM", "GradientSHAP"],
                            value="Attention Visualization",
                            label="🔬 Explanation Method",
                            info="Select the explainability technique to apply",
                        )

                        gr.Markdown("**Attention-specific Parameters:**")
                        with gr.Row():
                            layer_index = gr.Slider(
                                minimum=0,
                                maximum=11,
                                value=6,
                                step=1,
                                label="Layer Index",
                                info="Which transformer layer to visualize (0-11)",
                            )

                        with gr.Row():
                            head_index = gr.Slider(
                                minimum=0,
                                maximum=11,
                                value=0,
                                step=1,
                                label="Head Index",
                                info="Which attention head to visualize (0-11)",
                            )

                    analyze_btn = gr.Button("🚀 Analyze Image", variant="primary", size="lg")
                    status_output = gr.Textbox(
                        label="📊 Analysis Status",
                        interactive=False,
                        placeholder="Upload an image and click 'Analyze Image' to start...",
                        lines=4,
                        max_lines=6,
                    )

                with gr.Column(scale=2):
                    with gr.Row():
                        original_display = gr.Image(
                            label="📸 Processed Image", interactive=False, height=300
                        )
                        prediction_display = gr.Plot(label="📊 Top Predictions")

                    explanation_display = gr.Plot(label="🔍 Explanation Visualization")

            # Connect the analyze button
            analyze_btn.click(
                fn=analyze_image_basic,
                inputs=[image_input, model_choice, xai_method, layer_index, head_index],
                outputs=[original_display, prediction_display, explanation_display, status_output],
            )

        # Tab 2: Counterfactual Analysis
        with gr.TabItem("🔄 Counterfactual Analysis"):
            gr.Markdown(
                """
                ### Testing Model Robustness
                Systematically perturb image regions to understand which areas are most critical for predictions.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    cf_image_input = gr.Image(
                        label="📁 Upload Image",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=350,
                    )

                    with gr.Accordion("⚙️ Counterfactual Settings", open=True):
                        patch_size = gr.Slider(
                            minimum=16,
                            maximum=64,
                            value=32,
                            step=16,
                            label="🔲 Patch Size",
                            info="Size of perturbation patches - 16, 32, 48, or 64 pixels",
                        )

                        perturbation_type = gr.Dropdown(
                            choices=["blur", "blackout", "gray", "noise"],
                            value="blur",
                            label="🎨 Perturbation Type",
                            info="How to modify image patches",
                        )

                        gr.Markdown(
                            """
                        **Perturbation Types:**
                        - **Blur**: Gaussian blur effect
                        - **Blackout**: Replace with black pixels
                        - **Gray**: Convert to grayscale
                        - **Noise**: Add random noise
                        """
                        )

                    cf_analyze_btn = gr.Button(
                        "🔄 Run Counterfactual Analysis", variant="primary", size="lg"
                    )
                    cf_status_output = gr.Textbox(
                        label="📊 Analysis Status",
                        interactive=False,
                        placeholder="Upload an image and click to start counterfactual analysis...",
                        lines=5,
                        max_lines=8,
                    )

                with gr.Column(scale=2):
                    cf_explanation_display = gr.Plot(label="🔄 Counterfactual Analysis Results")

                    gr.Markdown(
                        """
                    **Understanding Results:**
                    - **Confidence Change**: How much the model's certainty shifts
                    - **Prediction Flip Rate**: Percentage of patches causing misclassification
                    - **Sensitive Regions**: Areas most critical to the model's decision
                    """
                    )

            cf_analyze_btn.click(
                fn=analyze_counterfactual,
                inputs=[cf_image_input, model_choice, patch_size, perturbation_type],
                outputs=[cf_explanation_display, cf_status_output],
            )

        # Tab 3: Confidence Calibration
        with gr.TabItem("📊 Confidence Calibration"):
            gr.Markdown(
                """
                ### Evaluating Prediction Reliability
                Assess whether the model's confidence scores accurately reflect the likelihood of correct predictions.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    cal_image_input = gr.Image(
                        label="📁 Upload Sample Image",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=350,
                    )

                    with gr.Accordion("⚙️ Calibration Settings", open=True):
                        n_bins = gr.Slider(
                            minimum=5,
                            maximum=20,
                            value=10,
                            step=1,
                            label="📊 Number of Bins",
                            info="Granularity of calibration analysis (5-20)",
                        )

                        gr.Markdown(
                            """
                        **Calibration Metrics:**
                        - **Perfect calibration**: Confidence matches accuracy
                        - **Overconfident**: High confidence, low accuracy
                        - **Underconfident**: Low confidence, high accuracy
                        """
                        )

                    cal_analyze_btn = gr.Button(
                        "📊 Analyze Calibration", variant="primary", size="lg"
                    )
                    cal_status_output = gr.Textbox(
                        label="📊 Analysis Status",
                        interactive=False,
                        placeholder="Upload an image and click to analyze calibration...",
                        lines=5,
                        max_lines=8,
                    )

                with gr.Column(scale=2):
                    cal_explanation_display = gr.Plot(label="📊 Calibration Analysis Results")

                    gr.Markdown(
                        """
                    **Interpreting Calibration:**
                    - A well-calibrated model's confidence should match its accuracy
                    - If the model predicts 80% confidence, it should be correct 80% of the time
                    - Large deviations indicate calibration issues requiring attention
                    """
                    )

            cal_analyze_btn.click(
                fn=analyze_calibration,
                inputs=[cal_image_input, model_choice, n_bins],
                outputs=[cal_explanation_display, cal_status_output],
            )

        # Tab 4: Bias Detection
        with gr.TabItem("⚖️ Bias Detection"):
            gr.Markdown(
                """
                ### Identifying Performance Disparities
                Detect potential biases by comparing model performance across different data subgroups.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    bias_image_input = gr.Image(
                        label="📁 Upload Sample Image",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=350,
                    )

                    gr.Markdown(
                        """
                    **Generated Subgroups:**
                    - Original image (baseline)
                    - Increased brightness
                    - Decreased brightness
                    - Enhanced contrast
                    """
                    )

                    bias_analyze_btn = gr.Button("⚖️ Detect Bias", variant="primary", size="lg")
                    bias_status_output = gr.Textbox(
                        label="📊 Analysis Status",
                        interactive=False,
                        placeholder="Upload an image and click to detect potential biases...",
                        lines=6,
                        max_lines=10,
                    )

                with gr.Column(scale=2):
                    bias_explanation_display = gr.Plot(label="⚖️ Bias Detection Results")

                    gr.Markdown(
                        """
                    **Understanding Bias Metrics:**
                    - Compare confidence scores across subgroups
                    - Large disparities may indicate systematic biases
                    - Consider demographic, environmental, and quality variations
                    - Use findings to improve data collection and model training
                    """
                    )

            bias_analyze_btn.click(
                fn=analyze_bias_detection,
                inputs=[bias_image_input, model_choice],
                outputs=[bias_explanation_display, bias_status_output],
            )

    # Footer
    gr.HTML(
        """
        <div style="
            margin-top: 3rem; 
            padding: 2rem;
            background: rgba(30, 41, 59, 0.3);
            border-top: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 16px;
            text-align: center;
        ">
            <p style="
                color: #64748b;
                font-size: 0.95rem;
                margin: 0;
            ">
                Built with ❤️ using <strong style="color: #a5b4fc;">Gradio</strong>, 
                <strong style="color: #c4b5fd;">Transformers</strong>, and 
                <strong style="color: #f9a8d4;">Captum</strong>
            </p>
            <p style="
                color: #475569;
                font-size: 0.85rem;
                margin-top: 0.5rem;
            ">
                © 2024 ViT Auditing Toolkit • For research and educational purposes
            </p>
        </div>
        """
    )

# Launch the application
if __name__ == "__main__":
    import os as _os
    # Use dynamic host/port for portability (e.g., Hugging Face Spaces)
    host = "0.0.0.0"
    port = int(_os.environ.get("PORT", "7860"))
    demo.launch(server_name=host, server_port=port, share=False, show_error=True)
