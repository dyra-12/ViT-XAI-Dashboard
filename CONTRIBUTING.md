# Contributing to ViT Auditing Toolkit

First off, thank you for considering contributing to the ViT Auditing Toolkit! It's people like you that make this tool better for everyone.

## 🌟 Ways to Contribute

### 1. Reporting Bugs 🐛

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Screenshots** if applicable
- **Environment details** (OS, Python version, etc.)

**Example:**
```markdown
**Bug**: GradCAM visualization fails with ViT-Large model

**Steps to reproduce:**
1. Select ViT-Large from dropdown
2. Upload any image
3. Select GradCAM method
4. Click "Analyze Image"

**Expected:** GradCAM heatmap visualization
**Actual:** Error message "AttributeError: ..."

**Environment:**
- OS: Ubuntu 22.04
- Python: 3.10.12
- PyTorch: 2.2.0
```

### 2. Suggesting Features ✨

Feature requests are welcome! Please provide:

- **Clear use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Additional context**: Screenshots, mockups, references

### 3. Contributing Code 💻

#### Development Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/ViT-XAI-Dashboard.git
cd ViT-XAI-Dashboard

# 3. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install development dependencies
pip install pytest black flake8 mypy

# 6. Create a feature branch
git checkout -b feature/amazing-feature
```

#### Code Style Guidelines

**Python Style:**
- Follow [PEP 8](https://pep8.org/)
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use meaningful variable names

**Formatting:**
```bash
# Format code with Black
black src/ tests/ app.py

# Check style with flake8
flake8 src/ tests/ app.py --max-line-length=100

# Type checking with mypy
mypy src/ --ignore-missing-imports
```

**Documentation:**
- Add docstrings to all functions and classes
- Use Google-style docstrings
- Update README.md if adding new features

**Example:**
```python
def explain_attention(model, processor, image, layer_index=6, head_index=0):
    """
    Extract and visualize attention weights from a specific layer and head.
    
    Args:
        model: Pre-trained ViT model with attention outputs enabled.
        processor: Image processor for model input preparation.
        image (PIL.Image): Input image to analyze.
        layer_index (int): Transformer layer index (0-11 for base model).
        head_index (int): Attention head index (0-11 for base model).
    
    Returns:
        matplotlib.figure.Figure: Visualization of attention patterns.
    
    Raises:
        ValueError: If layer_index or head_index is out of range.
        RuntimeError: If attention weights cannot be extracted.
    
    Example:
        >>> from PIL import Image
        >>> image = Image.open("cat.jpg")
        >>> fig = explain_attention(model, processor, image, layer_index=6)
        >>> fig.savefig("attention.png")
    """
    # Implementation...
```

#### Testing

All new features must include tests:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_explainer.py

# Run with coverage
pytest --cov=src tests/
```

**Writing Tests:**
```python
import pytest
from src.explainer import explain_attention

def test_attention_visualization():
    """Test attention visualization with valid inputs."""
    # Setup
    model, processor = load_test_model()
    image = create_test_image()
    
    # Execute
    fig = explain_attention(model, processor, image, layer_index=6)
    
    # Assert
    assert fig is not None
    assert len(fig.axes) > 0

def test_attention_invalid_layer():
    """Test attention visualization with invalid layer index."""
    model, processor = load_test_model()
    image = create_test_image()
    
    with pytest.raises(ValueError):
        explain_attention(model, processor, image, layer_index=99)
```

#### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(explainer): add LIME explainability method

- Implement LIME-based explanations
- Add visualization function
- Update documentation

Closes #42
```

```
fix(gradcam): resolve tensor dimension mismatch

GradCAM was failing for batch size != 1 due to
incorrect tensor reshaping. Now properly handles
single image inputs.

Fixes #38
```

#### Pull Request Process

1. **Update documentation**: README, docstrings, etc.
2. **Add tests**: Ensure your code is tested
3. **Run tests locally**: All tests must pass
4. **Update CHANGELOG**: Add your changes
5. **Create PR**: Use a clear title and description

**PR Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Tested manually with various inputs

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings or errors
- [ ] Commit messages are clear
```

### 4. Improving Documentation 📝

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples and tutorials
- Improve code comments
- Create video demonstrations
- Translate to other languages

### 5. Reviewing Pull Requests 👀

Help review open pull requests:

- Test the changes locally
- Provide constructive feedback
- Check for potential issues
- Verify documentation is updated

## 🎯 Good First Issues

Look for issues labeled `good first issue` or `help wanted` - these are great starting points!

## 📋 Project Priorities

Current focus areas:
1. **Stability**: Bug fixes and error handling
2. **Performance**: Optimization for large models
3. **Features**: Additional explainability methods
4. **Documentation**: More examples and tutorials
5. **Testing**: Improved test coverage

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all.

### Our Standards

**Positive behavior includes:**
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Harassment, trolling, or discriminatory comments
- Personal or political attacks
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to the project maintainers. All complaints will be reviewed and investigated.

## 📬 Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/dyra-12/ViT-XAI-Dashboard/discussions)
- **Bugs**: Open an [issue](https://github.com/dyra-12/ViT-XAI-Dashboard/issues)
- **Chat**: Join our community (link coming soon)

## 🙏 Thank You!

Your contributions, large or small, make this project better. We appreciate your time and effort!

---

**Happy Contributing! 🎉**
