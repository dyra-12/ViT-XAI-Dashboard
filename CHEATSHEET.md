# 🚀 ViT Auditing Toolkit - Quick Reference

## One-Liner Commands

```bash
# Quick start
python app.py

# Download sample images
python download_samples.py

# Run tests
pytest tests/ -v

# Run with Docker
docker-compose up

# Check code style
black --check src/ tests/ app.py

# Generate coverage report
pytest --cov=src --cov-report=html tests/
```

---

## 📂 Project Structure Quick Map

```
ViT-XAI-Dashboard/
├── app.py                          # 🎯 Main application - START HERE
├── requirements.txt                # 📦 Dependencies
│
├── src/                            # 🧠 Core functionality
│   ├── model_loader.py            # Load ViT models from HF
│   ├── predictor.py               # Make predictions
│   ├── explainer.py               # XAI methods (Attention, GradCAM, SHAP)
│   ├── auditor.py                 # Advanced auditing tools
│   └── utils.py                   # Helper functions
│
├── examples/                       # 🖼️ Test images (20 images)
│   ├── basic_explainability/      # For Tab 1
│   ├── counterfactual/           # For Tab 2
│   ├── calibration/              # For Tab 3
│   ├── bias_detection/           # For Tab 4
│   └── general/                  # Misc testing
│
├── tests/                         # 🧪 Unit tests
│   ├── test_phase1_complete.py   # Basic tests
│   └── test_advanced_features.py # Advanced tests
│
└── Documentation/                 # 📚 All docs
    ├── README.md                 # Main documentation
    ├── QUICKSTART.md            # 5-minute setup
    ├── TESTING.md               # Testing guide
    ├── CONTRIBUTING.md          # Dev guidelines
    └── PROJECT_SUMMARY.md       # This file
```

---

## 🎯 Common Tasks

### Start the Dashboard
```bash
python app.py
# Opens at http://localhost:7860
```

### Test a Single Tab
```bash
# 1. Start app: python app.py
# 2. Go to http://localhost:7860
# 3. Load ViT-Base model
# 4. Tab 1: Upload examples/basic_explainability/cat_portrait.jpg
# 5. Click "Analyze Image"
```

### Add New Test Image
```bash
# Option 1: Manual
cp /path/to/image.jpg examples/basic_explainability/

# Option 2: Download from URL
curl -L "https://example.com/image.jpg" -o examples/general/my_image.jpg
```

### Run Quick Test
```bash
# Smoke test (verify everything works)
python app.py &
sleep 10
curl http://localhost:7860
# If no error, you're good!
```

---

## 🔍 Tab Reference

### Tab 1: Basic Explainability (🔍)
**Purpose**: Understand predictions  
**Methods**: Attention, GradCAM, GradientSHAP  
**Best Images**: examples/basic_explainability/  
**Use When**: Want to see what model focuses on

### Tab 2: Counterfactual Analysis (🔄)
**Purpose**: Test robustness  
**Methods**: Patch perturbation (blur/blackout/gray/noise)  
**Best Images**: examples/counterfactual/  
**Use When**: Testing prediction stability

### Tab 3: Confidence Calibration (📊)
**Purpose**: Validate confidence scores  
**Methods**: Calibration curves, reliability diagrams  
**Best Images**: examples/calibration/  
**Use When**: Checking if confidence matches accuracy

### Tab 4: Bias Detection (⚖️)
**Purpose**: Find performance disparities  
**Methods**: Subgroup analysis  
**Best Images**: examples/bias_detection/  
**Use When**: Testing fairness across conditions

---

## 🎨 Customization Quick Tips

### Change Port
```python
# app.py, last line:
demo.launch(server_port=7860)  # Change 7860 to your port
```

### Add New Model
```python
# src/model_loader.py:
SUPPORTED_MODELS = {
    "ViT-Base": "google/vit-base-patch16-224",
    "ViT-Large": "google/vit-large-patch16-224",
    "Your-Model": "your-username/your-vit-model",  # Add this
}
```

### Modify Colors
```python
# app.py, custom_css variable:
# Change gradient colors, backgrounds, etc.
```

---

## 🐛 Troubleshooting Quick Fixes

### Port Already in Use
```bash
# Linux/Mac:
lsof -ti:7860 | xargs kill -9
# Windows:
netstat -ano | findstr :7860
taskkill /PID <PID> /F
```

### Out of Memory
```python
# Use smaller model
model_choice = "ViT-Base"  # instead of ViT-Large

# Or clear GPU cache
import torch
torch.cuda.empty_cache()
```

### Model Download Fails
```bash
# Set cache directory
export HF_HOME="/path/to/writable/dir"
export TRANSFORMERS_CACHE="/path/to/writable/dir"
```

### Slow Inference
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version if False
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Model Comparison

| Feature | ViT-Base | ViT-Large |
|---------|----------|-----------|
| Parameters | 86M | 304M |
| Memory | ~2GB | ~4GB |
| Speed | Faster | Slower |
| Accuracy | ~81% | ~83% |
| Best For | Quick tests | Production |

---

## 🧪 Testing Shortcuts

### Minimal Test (30 seconds)
```bash
python app.py &
# Load model → Upload cat_portrait.jpg → Analyze
```

### Full Test (5 minutes)
```bash
# One image per tab
Tab 1: cat_portrait.jpg
Tab 2: flower.jpg
Tab 3: clear_panda.jpg
Tab 4: dog_daylight.jpg
```

### Comprehensive Test (30 minutes)
```bash
# Follow TESTING.md for all 22 tests
```

---

## 📚 Documentation Quick Links

- **Setup**: QUICKSTART.md
- **Testing**: TESTING.md
- **Contributing**: CONTRIBUTING.md
- **Full Docs**: README.md
- **This Guide**: PROJECT_SUMMARY.md

---

## 🔗 Useful URLs

```bash
# Local
http://localhost:7860              # Main app
http://localhost:7860/docs         # API docs (if enabled)

# Hugging Face (after deployment)
https://huggingface.co/spaces/YOUR-USERNAME/vit-auditing-toolkit

# GitHub (your repo)
https://github.com/dyra-12/ViT-XAI-Dashboard
```

---

## ⌨️ Keyboard Shortcuts (Browser)

- `Ctrl/Cmd + R`: Reload interface
- `Ctrl/Cmd + Shift + I`: Open dev tools
- `Ctrl/Cmd + K`: Clear console

---

## 📦 File Sizes Reference

```
Total Project: ~1.6 MB
├── Code: ~200 KB
├── Images: ~1.3 MB
├── Docs: ~100 KB
└── Config: ~10 KB
```

---

## 🎯 Performance Benchmarks

**Typical Response Times**:
- Model Loading: 5-15s (first time)
- Prediction: 0.5-2s
- Attention Viz: 1-3s
- GradCAM: 2-4s
- GradientSHAP: 8-15s
- Counterfactual: 10-30s
- Calibration: 5-10s
- Bias Detection: 5-10s

---

## 💡 Pro Tips

1. **Use ViT-Base** for quick testing
2. **Use ViT-Large** for production/demos
3. **Cache results** if analyzing same image repeatedly
4. **Start with Tab 1** to understand predictions
5. **Use examples/** images for consistent testing
6. **Check TESTING.md** for detailed test cases
7. **Read CONTRIBUTING.md** before making changes

---

## 🆘 Getting Help

1. Check this file first
2. Read relevant documentation
3. Search GitHub issues
4. Open new issue with details
5. Join discussions

---

## ✅ Pre-Demo Checklist

Before showing to others:

- [ ] App runs without errors
- [ ] All tabs functional
- [ ] Sample images loaded
- [ ] Model loads quickly
- [ ] UI looks professional
- [ ] No console errors
- [ ] README updated with your info

---

**Keep this file handy for quick reference! 📌**

*Last updated: October 26, 2024*
