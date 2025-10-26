# 🖼️ Example Images for Testing

This directory contains sample images for testing the ViT Auditing Toolkit across different analysis types.

## 📁 Directory Structure

```
examples/
├── basic_explainability/    # Images for testing prediction and explanation
├── counterfactual/          # Images for robustness testing
├── calibration/             # Images for confidence calibration
├── bias_detection/          # Images for bias analysis
└── general/                 # General test images
```

## 🎯 Recommended Test Images by Tab

### Tab 1: Basic Explainability (🔍)
**Purpose**: Test prediction accuracy and explanation quality

**Recommended Images**:
- **Clear single objects**: Cat, dog, car, bird (high confidence predictions)
- **Complex scenes**: Multiple objects, cluttered backgrounds
- **Ambiguous images**: Similar classes (husky vs wolf, muffin vs chihuahua)
- **Different angles**: Top view, side view, close-up

**Examples to add**:
```
basic_explainability/
├── cat_portrait.jpg          # Clear cat face
├── dog_playing.jpg           # Dog in action
├── bird_flying.jpg           # Bird in flight
├── car_sports.jpg            # Sports car
├── multiple_objects.jpg      # Complex scene
├── ambiguous_animal.jpg      # Hard to classify
└── unusual_angle.jpg         # Non-standard viewpoint
```

### Tab 2: Counterfactual Analysis (🔄)
**Purpose**: Test prediction robustness and identify critical regions

**Recommended Images**:
- **Simple backgrounds**: Easy to see perturbation effects
- **Centered objects**: Better for patch analysis
- **Distinct features**: Eyes, wheels, wings (test if they're critical)
- **Varying complexity**: Simple to complex objects

**Examples to add**:
```
counterfactual/
├── face_centered.jpg         # Test facial feature importance
├── car_side_view.jpg         # Test wheel/door importance
├── building_architecture.jpg # Test structural elements
├── simple_object.jpg         # Baseline robustness test
└── textured_object.jpg       # Test texture vs shape
```

### Tab 3: Confidence Calibration (📊)
**Purpose**: Test if model confidence matches accuracy

**Recommended Images**:
- **High quality**: Should have high confidence
- **Low quality**: Blurry, dark, pixelated
- **Edge cases**: Partial objects, occluded views
- **Various difficulties**: Easy to hard classifications

**Examples to add**:
```
calibration/
├── clear_high_quality.jpg    # Should be high confidence
├── slightly_blurry.jpg       # Medium confidence expected
├── very_blurry.jpg           # Low confidence expected
├── dark_lighting.jpg         # Test lighting robustness
├── partial_object.jpg        # Occluded/cropped
└── mixed_quality_set/        # Batch of varied quality
```

### Tab 4: Bias Detection (⚖️)
**Purpose**: Detect performance variations across subgroups

**Recommended Images**:
- **Same subject, different conditions**: Lighting, weather, seasons
- **Demographic variations**: Different breeds, ages, sizes
- **Environmental context**: Indoor vs outdoor, urban vs rural
- **Quality variations**: Professional vs amateur photos

**Examples to add**:
```
bias_detection/
├── day_lighting.jpg          # Same scene in daylight
├── night_lighting.jpg        # Same scene at night
├── sunny_weather.jpg         # Clear conditions
├── rainy_weather.jpg         # Poor conditions
├── indoor_scene.jpg          # Controlled environment
├── outdoor_scene.jpg         # Natural environment
└── subgroup_sets/            # Organized by demographic
    ├── lighting/
    ├── weather/
    ├── quality/
    └── environment/
```

## 🌐 Where to Get Test Images

### Free Image Sources (Royalty-Free)

1. **Unsplash** (https://unsplash.com)
   - High quality, free to use
   - Good for professional-looking tests
   ```bash
   # Example downloads
   curl -L "https://unsplash.com/photos/[photo-id]/download" -o image.jpg
   ```

2. **Pexels** (https://www.pexels.com)
   - Free stock photos and videos
   - Good variety of subjects

3. **Pixabay** (https://pixabay.com)
   - Free images and videos
   - Commercial use allowed

4. **ImageNet Sample** (https://image-net.org)
   - Validation set samples
   - Directly relevant to ViT training

### Quick Download Scripts

#### Download Sample Images
```bash
# Create directories
mkdir -p examples/{basic_explainability,counterfactual,calibration,bias_detection,general}

# Download sample cat image
curl -L "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800" \
  -o examples/basic_explainability/cat_portrait.jpg

# Download sample dog image
curl -L "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800" \
  -o examples/basic_explainability/dog_portrait.jpg

# Download sample bird image
curl -L "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=800" \
  -o examples/basic_explainability/bird_flying.jpg

# Download sample car image
curl -L "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=800" \
  -o examples/basic_explainability/sports_car.jpg
```

#### Use Your Own Images
```bash
# Simply copy your images to the appropriate directory
cp /path/to/your/image.jpg examples/basic_explainability/
```

## 📋 Image Requirements

### Technical Specifications
- **Format**: JPG, PNG, WebP
- **Size**: Any size (will be resized to 224×224)
- **Color**: RGB (grayscale will be converted)
- **Quality**: Higher quality = better analysis

### Recommended Guidelines
- **Resolution**: At least 224×224 pixels (higher is fine)
- **Aspect Ratio**: Any (will be center-cropped)
- **File Size**: < 10MB for faster upload
- **Content**: Clear, well-lit subjects work best

## 🧪 Testing Checklist

### Basic Testing
- [ ] Upload works for all image formats (JPG, PNG)
- [ ] Predictions are reasonable
- [ ] Visualizations render correctly
- [ ] Interface is responsive

### Tab-Specific Testing

#### Basic Explainability
- [ ] Attention maps show relevant regions
- [ ] GradCAM highlights correctly
- [ ] SHAP values make sense
- [ ] All layers/heads accessible

#### Counterfactual Analysis
- [ ] Perturbations are visible
- [ ] Sensitivity maps are informative
- [ ] All perturbation types work
- [ ] Metrics are calculated

#### Confidence Calibration
- [ ] Calibration curves render
- [ ] Metrics are reasonable
- [ ] Bin settings work correctly

#### Bias Detection
- [ ] Subgroups are compared
- [ ] Variations are generated
- [ ] Metrics show differences

## 💡 Tips for Good Test Images

### Do's ✅
- Use clear, well-lit images
- Test with ImageNet classes the model knows
- Try edge cases and challenging examples
- Test with images from different sources
- Use consistent naming conventions

### Don'ts ❌
- Don't use copyrighted images (use free sources)
- Don't use extremely large files (> 50MB)
- Don't use corrupted or invalid image files
- Don't rely on a single image type

## 🎯 Creating Your Own Test Set

```bash
#!/bin/bash
# Script to organize your test images

# Create structure
mkdir -p examples/{basic_explainability,counterfactual,calibration,bias_detection}

# Organize by category
echo "Organizing images..."

# Move or copy your images to appropriate folders
# Rename for consistency
mv unclear_image.jpg examples/basic_explainability/01_cat.jpg
mv another_image.jpg examples/basic_explainability/02_dog.jpg

echo "✅ Test image set ready!"
```

## 📊 ImageNet Classes Reference

Common classes the ViT models can recognize (examples):

- **Animals**: cat, dog, bird, fish, horse, elephant, bear, tiger, etc.
- **Vehicles**: car, truck, bus, motorcycle, bicycle, airplane, boat, etc.
- **Objects**: chair, table, bottle, cup, keyboard, phone, book, etc.
- **Nature**: tree, flower, mountain, beach, forest, etc.
- **Food**: pizza, burger, cake, fruit, vegetables, etc.

See full list: https://github.com/anishathalye/imagenet-simple-labels

## 🔗 Quick Links

- **Unsplash API**: https://unsplash.com/developers
- **Pexels API**: https://www.pexels.com/api/
- **ImageNet**: https://image-net.org/
- **COCO Dataset**: https://cocodataset.org/

---

**Ready to test?** Add your images to the appropriate directories and start analyzing! 🚀
