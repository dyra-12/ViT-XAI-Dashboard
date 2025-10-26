# Changelog

All notable changes to the ViT Auditing Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Support for additional ViT variants (DeiT, BEiT, Swin Transformer)
- Batch processing capabilities
- Export functionality for reports
- Custom model upload support
- API endpoints for programmatic access

## [1.0.0] - 2024-10-26

### Added
- Initial release of ViT Auditing Toolkit
- Basic Explainability features:
  - Attention Visualization with layer/head selection
  - GradCAM implementation using Captum
  - GradientSHAP for pixel-level attribution
- Advanced Auditing features:
  - Counterfactual Analysis with patch perturbation
  - Confidence Calibration analysis
  - Bias Detection across subgroups
- Web interface using Gradio:
  - Modern, responsive UI with custom styling
  - Four-tab interface for different analysis types
  - Real-time visualization of results
- Model support:
  - ViT-Base (google/vit-base-patch16-224)
  - ViT-Large (google/vit-large-patch16-224)
- Comprehensive documentation:
  - Detailed README with usage guides
  - Technical explanations of methods
  - Installation instructions
- Testing suite:
  - Unit tests for core functionality
  - Integration tests for advanced features
- Docker support for easy deployment
- CI/CD pipeline with GitHub Actions

### Technical Details
- PyTorch 2.2+ compatibility
- Hugging Face Transformers integration
- Captum for model interpretability
- Gradio 4.19+ for web interface
- Matplotlib for visualizations

### Documentation
- Comprehensive README.md
- Contributing guidelines
- MIT License
- Code of conduct

## [0.1.0] - 2024-10-15

### Added
- Project initialization
- Basic project structure
- Core module implementations
- Initial model loading functionality

---

## Version History

### Version Numbering
- **Major version (X.0.0)**: Incompatible API changes
- **Minor version (0.X.0)**: New functionality, backwards-compatible
- **Patch version (0.0.X)**: Backwards-compatible bug fixes

### Release Notes Format
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

---

For more details on any version, see the [GitHub Releases](https://github.com/dyra-12/ViT-XAI-Dashboard/releases) page.
