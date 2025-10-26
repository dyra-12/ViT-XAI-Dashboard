# 🚀 Quick Start Guide

Get up and running with the ViT Auditing Toolkit in under 5 minutes!

## 📋 Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher installed
- pip package manager
- (Optional) CUDA-compatible GPU for faster inference

Check your Python version:
```bash
python --version  # Should be 3.8+
```

## ⚡ Installation Options

### Option 1: Quick Install (Recommended for Most Users)

```bash
# Clone the repository
git clone https://github.com/dyra-12/ViT-XAI-Dashboard.git
cd ViT-XAI-Dashboard

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Open your browser to `http://localhost:7860` 🎉

### Option 2: Virtual Environment (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/dyra-12/ViT-XAI-Dashboard.git
cd ViT-XAI-Dashboard

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Option 3: Docker (Production Ready)

```bash
# Build the image
docker build -t vit-auditing-toolkit .

# Run the container
docker run -p 7860:7860 vit-auditing-toolkit

# Or use docker-compose
docker-compose up
```

### Option 4: Google Colab (No Installation Required)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dyra-12/ViT-XAI-Dashboard/blob/main/notebooks/colab_demo.ipynb)

Run directly in your browser with free GPU access!

## 🎯 First-Time Usage

### Step 1: Launch the Application

```bash
python app.py
```

You should see:
```
✅ Model and processor loaded successfully on cpu!
Running on local URL:  http://localhost:7860
```

### Step 2: Open the Dashboard

Open your web browser and navigate to:
```
http://localhost:7860
```

### Step 3: Load a Model

1. In the **"Select Model"** dropdown, choose `ViT-Base`
2. Click the **"🔄 Load Model"** button
3. Wait for the confirmation: `✅ Model loaded: google/vit-base-patch16-224`

### Step 4: Analyze Your First Image

#### Option A: Use a Sample Image
1. Download a sample image:
   ```bash
   curl -o sample.jpg https://images.unsplash.com/photo-1574158622682-e40e69881006
   ```
2. Or use any image from your computer

#### Option B: Try the Demo Images
We provide sample images in the `examples/` directory:
- `examples/cat.jpg` - Cat portrait
- `examples/dog.jpg` - Dog portrait  
- `examples/bird.jpg` - Bird in flight
- `examples/car.jpg` - Sports car

### Step 5: Run Your First Analysis

1. Go to the **"🔍 Basic Explainability"** tab
2. Click **"📁 Upload Image"** and select your image
3. Keep default settings (Attention Visualization, Layer 6, Head 0)
4. Click **"🚀 Analyze Image"**
5. View the results:
   - **Processed Image**: Your input image
   - **Top Predictions**: Bar chart of confidence scores
   - **Explanation Visualization**: Attention heatmap

## 🎓 Learning Path

### Beginner: Understanding Predictions
Start with **Basic Explainability** to see:
- What objects the model recognizes
- Which image regions are most important
- How confident the model is

**Try this:**
```
1. Upload a clear photo of a single object
2. Use Attention Visualization (default)
3. Try different layers (0-11) to see how features evolve
4. Switch to GradCAM for a different perspective
```

### Intermediate: Testing Robustness
Move to **Counterfactual Analysis** to explore:
- How stable are predictions when parts of the image change?
- Which regions are critical vs. irrelevant?

**Try this:**
```
1. Upload the same image from before
2. Start with patch_size=32, perturbation_type="blur"
3. Click "Run Counterfactual Analysis"
4. Try different perturbation types to see variations
```

### Advanced: Model Validation
Use **Confidence Calibration** and **Bias Detection**:
- Is the model overconfident?
- Does performance vary across different image conditions?

**Try this:**
```
1. Test calibration with various images
2. Check if confidence matches actual reliability
3. Use bias detection to compare subgroups
```

## 💡 Common Use Cases

### Use Case 1: Debugging Misclassifications

**Problem**: Model misclassifies your image  
**Solution**: Use Basic Explainability to see what it's looking at

```python
# Steps:
1. Upload misclassified image
2. Check top predictions (might be close to correct class)
3. View attention maps - is it focusing on the right region?
4. Try GradCAM to see discriminative regions
5. Use counterfactual analysis to find sensitive areas
```

### Use Case 2: Model Selection

**Problem**: Choosing between ViT-Base and ViT-Large  
**Solution**: Compare predictions and confidence

```python
# Steps:
1. Load ViT-Base, analyze your image
2. Note confidence scores and predictions
3. Load ViT-Large, analyze same image
4. Compare:
   - Prediction accuracy
   - Confidence levels
   - Attention patterns
   - Inference time
```

### Use Case 3: Dataset Quality Check

**Problem**: Ensuring your dataset is suitable for ViT  
**Solution**: Use bias detection and calibration

```python
# Steps:
1. Sample random images from dataset
2. Run bias detection to check for systematic issues
3. Check calibration to see if model is overconfident
4. Identify problematic image categories
```

## 🔧 Troubleshooting

### Problem: Out of Memory Error

**Solution:**
```bash
# Use ViT-Base instead of ViT-Large
# Or reduce image batch size
# Or close other applications

# For programmatic use:
import torch
torch.cuda.empty_cache()  # Clear GPU memory
```

### Problem: Slow Inference

**Solution:**
```bash
# Check if using GPU:
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Model Download Fails

**Solution:**
```bash
# Set Hugging Face cache directory:
export HF_HOME="/path/to/writable/directory"

# Or download manually:
python -c "from transformers import ViTImageProcessor; ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')"
```

### Problem: Port Already in Use

**Solution:**
```bash
# Use a different port:
# Modify app.py, line: demo.launch(server_port=7861)

# Or kill the process using port 7860:
# Linux/Mac:
lsof -ti:7860 | xargs kill -9

# Windows:
netstat -ano | findstr :7860
taskkill /PID <PID> /F
```

## 📊 Example Workflows

### Workflow 1: Quick Image Classification

```bash
# 1. Start application
python app.py

# 2. In browser (http://localhost:7860):
#    - Load ViT-Base model
#    - Upload image
#    - Click "Analyze Image"
#    - View top predictions

# Total time: < 1 minute
```

### Workflow 2: Comprehensive Model Audit

```bash
# 1. Start application
python app.py

# 2. For each test image:
#    Tab 1: Check predictions and attention
#    Tab 2: Test robustness with perturbations
#    Tab 3: Validate confidence calibration
#    Tab 4: Check for bias across variations

# 3. Document findings
# 4. Iterate on model/data as needed

# Total time: 5-10 minutes per image
```

### Workflow 3: Research Experiment

```bash
# 1. Collect dataset of test images
# 2. For each explainability method:
#    - Run on all test images
#    - Export visualizations
#    - Compute metrics
# 3. Compare methods quantitatively
# 4. Generate paper figures

# Total time: Varies by dataset size
```

## 🎯 Next Steps

After completing this quick start:

1. **Explore Advanced Features**: Try all four tabs with different images
2. **Read Technical Docs**: Understand the methods in detail
3. **Customize Settings**: Adjust parameters for your use case
4. **Integrate into Workflow**: Use programmatically or via API
5. **Contribute**: Share improvements or report issues

## 📚 Additional Resources

- **Full Documentation**: [README.md](README.md)
- **API Reference**: [docs/api.md](docs/api.md)
- **Video Tutorials**: [YouTube Playlist](#)
- **Example Notebooks**: [notebooks/](notebooks/)
- **Community Forum**: [GitHub Discussions](https://github.com/dyra-12/ViT-XAI-Dashboard/discussions)

## 🆘 Getting Help

- **Issues**: [Report bugs](https://github.com/dyra-12/ViT-XAI-Dashboard/issues)
- **Discussions**: [Ask questions](https://github.com/dyra-12/ViT-XAI-Dashboard/discussions)
- **Email**: dyra12@example.com

---

**Ready to dive deeper?** Check out the [full documentation](README.md) or [contributing guidelines](CONTRIBUTING.md)!

🎉 **Happy Exploring!** 🎉
