# 🎯 ViT Auditing Toolkit

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.19+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A Comprehensive Explainability and Validation Dashboard for Vision Transformers**

[🚀 Live Demo](#) | [📖 Documentation](#features) | [💡 Examples](#usage-guide) | [🤝 Contributing](#contributing)

<img src="https://via.placeholder.com/800x400/0f1419/6366f1?text=ViT+Auditing+Toolkit+Dashboard" alt="Dashboard Preview" width="800"/>

</div>

---

## 🌟 Overview

The **ViT Auditing Toolkit** is an advanced, interactive dashboard designed to help researchers, ML practitioners, and AI auditors understand, validate, and improve Vision Transformer (ViT) models. It provides a comprehensive suite of explainability techniques and auditing tools through an intuitive web interface.

### 🎭 Why This Toolkit?

- **🔍 Transparency**: Understand what your ViT models actually "see" and learn
- **✅ Validation**: Verify model reliability through systematic testing
- **⚖️ Fairness**: Detect potential biases across different data subgroups  
- **🛡️ Robustness**: Test prediction stability under various perturbations
- **📊 Calibration**: Ensure confidence scores reflect true prediction accuracy

---

## ✨ Features

### 🔬 Basic Explainability
Visualize and understand model predictions through multiple state-of-the-art techniques:

- **🎨 Attention Visualization**: See which image patches the transformer focuses on at each layer and head
- **🔥 GradCAM**: Gradient-weighted Class Activation Mapping for highlighting discriminative regions
- **💫 GradientSHAP**: Shapley value-based attribution for pixel-level importance

### 🔄 Counterfactual Analysis
Test model robustness by systematically perturbing image regions:

- **Patch Perturbation**: Apply blur, blackout, grayscale, or noise to image patches
- **Sensitivity Mapping**: Identify which regions are critical for predictions
- **Prediction Stability**: Measure confidence changes and prediction flip rates

### 📊 Confidence Calibration
Evaluate whether model confidence scores accurately reflect prediction reliability:

- **Calibration Curves**: Visual assessment of confidence vs accuracy alignment
- **Reliability Diagrams**: Binned analysis of prediction calibration
- **Metrics Dashboard**: Mean confidence, overconfidence rate, and underconfidence rate

### ⚖️ Bias Detection
Identify performance disparities across different data subgroups:

- **Subgroup Analysis**: Compare performance across demographic or environmental variations
- **Fairness Metrics**: Detect systematic biases in model predictions
- **Comparative Visualization**: Side-by-side analysis of confidence distributions

---

## 🚀 Live Demo

Try the toolkit instantly on Hugging Face Spaces:

### 👉 [Launch Interactive Demo](https://huggingface.co/spaces/YOUR-USERNAME/vit-auditing-toolkit)

*No installation required! Upload an image and start exploring.*

---

## 📸 Screenshots

<div align="center">

### Basic Explainability Interface
<img src="https://via.placeholder.com/700x400/1a1f2e/a5b4fc?text=Attention+Visualization+%26+Predictions" alt="Basic Explainability" width="700"/>

### Counterfactual Analysis
<img src="https://via.placeholder.com/700x400/1a1f2e/c4b5fd?text=Patch+Perturbation+Analysis" alt="Counterfactual Analysis" width="700"/>

### Calibration & Bias Detection
<img src="https://via.placeholder.com/700x400/1a1f2e/f9a8d4?text=Calibration+%26+Bias+Metrics" alt="Advanced Auditing" width="700"/>

</div>

---

## 🎯 Usage Guide

### Quick Start (3 Steps)

1. **Select a Model**: Choose between ViT-Base or ViT-Large from the dropdown
2. **Upload Your Image**: Any image you want to analyze (JPG, PNG, etc.)
3. **Choose Analysis Type**: Select from 4 tabs based on your needs

### Detailed Workflow

#### 🔍 For Understanding Predictions:
```
1. Go to "Basic Explainability" tab
2. Upload your image
3. Select explanation method (Attention/GradCAM/SHAP)
4. Adjust layer/head indices if needed
5. Click "Analyze Image"
6. View predictions and visual explanations
```

#### 🔄 For Testing Robustness:
```
1. Go to "Counterfactual Analysis" tab
2. Upload your image
3. Set patch size (16-64 pixels)
4. Choose perturbation type (blur/blackout/gray/noise)
5. Click "Run Analysis"
6. Review sensitivity maps and metrics
```

#### 📊 For Validating Confidence:
```
1. Go to "Confidence Calibration" tab
2. Upload a sample image
3. Adjust number of bins for analysis
4. Click "Analyze Calibration"
5. Review calibration curves and metrics
```

#### ⚖️ For Detecting Bias:
```
1. Go to "Bias Detection" tab
2. Upload a sample image
3. Click "Detect Bias"
4. Compare performance across generated subgroups
5. Review fairness metrics
```

---

## 💻 Local Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for faster inference)
- 8GB+ RAM

### Step 1: Clone the Repository

```bash
git clone https://github.com/dyra-12/ViT-XAI-Dashboard.git
cd ViT-XAI-Dashboard
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n vit-audit python=3.10
conda activate vit-audit
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python app.py
```

The dashboard will be available at `http://localhost:7860`

### 🐳 Docker Installation (Alternative)

```bash
# Build the Docker image
docker build -t vit-auditing-toolkit .

# Run the container
docker run -p 7860:7860 vit-auditing-toolkit
```

---

## 🏗️ Project Structure

```
ViT-XAI-Dashboard/
│
├── app.py                      # Main Gradio application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/
│   ├── __init__.py
│   ├── model_loader.py         # ViT model loading from Hugging Face
│   ├── predictor.py            # Prediction and classification logic
│   ├── explainer.py            # XAI methods (Attention, GradCAM, SHAP)
│   ├── auditor.py              # Advanced auditing tools
│   └── utils.py                # Helper functions and preprocessing
│
└── tests/
    ├── test_phase1_complete.py # Basic functionality tests
    └── test_advanced_features.py # Advanced auditing tests
```

---

## 🧠 Technical Details

### Vision Transformers (ViT)

Vision Transformers apply the transformer architecture (originally designed for NLP) to computer vision tasks. Key concepts:

- **Patch Embedding**: Images are split into fixed-size patches (e.g., 16×16 pixels)
- **Self-Attention**: Each patch attends to all other patches to capture global context
- **Layer Hierarchy**: Multiple transformer layers progressively refine representations
- **Classification Token**: A special [CLS] token aggregates information for final prediction

**Advantages:**
- Strong performance on large-scale datasets
- Captures long-range dependencies better than CNNs
- More interpretable through attention mechanisms

### Explainability Techniques

#### 1. Attention Visualization
**Method**: Direct visualization of transformer attention weights  
**Purpose**: Shows which image patches the model focuses on  
**Implementation**: Extracts attention matrices from specified layers/heads

```python
# Example: Layer 6, Head 0 typically captures semantic patterns
attention_map = model.encoder.layer[6].attention.self.attention_weights
```

#### 2. GradCAM (Gradient-weighted Class Activation Mapping)
**Method**: Uses gradients flowing into the final conv layer  
**Purpose**: Highlights discriminative regions for target class  
**Implementation**: Via Captum's `LayerGradCam`

```python
# Generates heatmap showing which regions support the prediction
gradcam = LayerGradCam(model, target_layer)
attribution = gradcam.attribute(input, target=predicted_class)
```

#### 3. GradientSHAP (Gradient-based Shapley Values)
**Method**: Combines Shapley values with gradient information  
**Purpose**: Pixel-level attribution with theoretical guarantees  
**Implementation**: Via Captum's `GradientShap`

```python
# Computes fair attribution using random baselines
gradient_shap = GradientShap(model)
attributions = gradient_shap.attribute(input, baselines=random_baselines)
```

### Auditing Methodologies

#### Counterfactual Analysis
Systematically modifies image regions to test:
- **Robustness**: Does the prediction remain stable?
- **Feature Importance**: Which regions matter most?
- **Adversarial Vulnerability**: How easy is it to fool the model?

#### Confidence Calibration
Measures alignment between predicted confidence and actual accuracy:
- **Well-calibrated**: 80% confidence → 80% correct
- **Overconfident**: 90% confidence → 60% correct (problem!)
- **Underconfident**: 50% confidence → 80% correct (less critical)

#### Bias Detection
Compares performance across subgroups to identify:
- **Demographic bias**: Different accuracy for different groups
- **Environmental bias**: Performance varies with lighting, quality, etc.
- **Systematic patterns**: Consistent over/under-performance

---

## 🔧 Supported Models

Currently supported Vision Transformer models from Hugging Face:

| Model | Parameters | Input Size | Accuracy (ImageNet) |
|-------|-----------|------------|---------------------|
| `google/vit-base-patch16-224` | 86M | 224×224 | ~81.3% |
| `google/vit-large-patch16-224` | 304M | 224×224 | ~82.6% |

**Easy to extend**: Add any Hugging Face ViT model to `src/model_loader.py`

---

## 📦 Dependencies

### Core Libraries

- **PyTorch** (≥2.2.0): Deep learning framework
- **Transformers** (≥4.35.0): Hugging Face model hub
- **Gradio** (≥4.19.0): Web interface framework
- **Captum** (≥0.7.0): Model interpretability library

### Supporting Libraries

- **Pillow**: Image processing
- **Matplotlib**: Visualization
- **NumPy**: Numerical computations

See `requirements.txt` for complete list with version constraints.

---

## 🎓 Use Cases

### Research
- **Interpretability Studies**: Analyze transformer attention patterns
- **Benchmark Explainability**: Compare XAI methods systematically
- **Model Auditing**: Validate models before deployment

### Industry
- **Model Validation**: Ensure reliability before production
- **Bias Auditing**: Detect and mitigate fairness issues
- **Regulatory Compliance**: Document model decision-making

### Education
- **Teaching Tool**: Demonstrate XAI concepts interactively
- **Student Projects**: Foundation for ML course assignments
- **Research Training**: Hands-on experience with modern techniques

---

## 🛣️ Roadmap

### Upcoming Features
- [ ] Support for additional ViT variants (DeiT, BEiT, Swin Transformer)
- [ ] Batch processing for multiple images
- [ ] Export functionality for reports and visualizations
- [ ] Custom model upload support
- [ ] Comparative analysis across multiple models
- [ ] Integration with model monitoring platforms
- [ ] Advanced bias metrics (demographic parity, equalized odds)
- [ ] Adversarial robustness testing
- [ ] API endpoint for programmatic access

### Long-term Vision
- Multi-modal transformer support (CLIP, ViLT)
- Video analysis capabilities
- Automated auditing pipelines
- Integration with MLOps platforms

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

1. **🐛 Bug Reports**: Open an issue with detailed reproduction steps
2. **✨ Feature Requests**: Suggest new explainability methods or auditing tools
3. **📝 Documentation**: Improve guides, add examples, fix typos
4. **💻 Code**: Submit pull requests for new features or fixes
5. **🎨 UI/UX**: Enhance the dashboard design and user experience

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/ViT-XAI-Dashboard.git
cd ViT-XAI-Dashboard

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python -m pytest tests/

# Commit and push
git commit -m "Add: your feature description"
git push origin feature/your-feature-name

# Open a pull request
```

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new features

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 ViT Auditing Toolkit Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full license text...]
```

---

## 📚 References & Citations

### Academic Papers

1. **Vision Transformers**  
   Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*.

2. **GradCAM**  
   Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV 2017*.

3. **SHAP**  
   Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS 2017*.

4. **Model Calibration**  
   Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML 2017*.

### Related Tools

- [Captum](https://captum.ai/): Model interpretability for PyTorch
- [Hugging Face Transformers](https://huggingface.co/transformers/): State-of-the-art NLP and Vision models
- [Gradio](https://gradio.app/): Fast ML demo creation

### Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{vit_auditing_toolkit_2024,
  title={ViT Auditing Toolkit: Comprehensive Explainability for Vision Transformers},
  author={dyra-12},
  year={2024},
  url={https://github.com/dyra-12/ViT-XAI-Dashboard}
}
```

---

## 🙏 Acknowledgments

- **Hugging Face** for providing pre-trained ViT models and the Transformers library
- **Captum Team** for the excellent interpretability library
- **Gradio Team** for the intuitive ML interface framework
- **PyTorch Community** for the robust deep learning ecosystem
- All contributors and users who provide feedback and improvements

---

## 📧 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/dyra-12/ViT-XAI-Dashboard/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/dyra-12/ViT-XAI-Dashboard/discussions)
- **Email**: dyra12@example.com

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐️ on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=dyra-12/ViT-XAI-Dashboard&type=Date)](https://star-history.com/#dyra-12/ViT-XAI-Dashboard&Date)

---

<div align="center">

**Built with ❤️ by the community**

[⬆ Back to Top](#-vit-auditing-toolkit)

</div>
