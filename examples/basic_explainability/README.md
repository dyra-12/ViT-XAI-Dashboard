# Basic Explainability Test Images

This folder contains images optimized for testing prediction and explanation quality.

## 📸 Recommended Images

### What to Include:
1. **Clear Single Objects**: Cat, dog, car, bird
2. **Complex Scenes**: Multiple objects, cluttered backgrounds  
3. **Ambiguous Cases**: Similar classes (husky vs wolf)
4. **Different Angles**: Top, side, close-up views

### Current Images:
- `cat_portrait.jpg` - Clear cat face for attention testing
- `dog_portrait.jpg` - Dog portrait for GradCAM
- `bird_flying.jpg` - Action shot for dynamic features
- `sports_car.jpg` - Vehicle with distinct features
- `coffee_cup.jpg` - Common object test

## 🧪 Testing Guide

### Test Attention Visualization:
```
1. Upload cat_portrait.jpg
2. Try different layers (0, 6, 11)
3. Observe how attention evolves
```

### Test GradCAM:
```
1. Upload sports_car.jpg
2. Select GradCAM method
3. Check if wheels/body are highlighted
```

### Test GradientSHAP:
```
1. Upload bird_flying.jpg
2. Select GradientSHAP
3. Verify wing/head importance
```

## 💡 Tips
- Use high-resolution images (> 224px)
- Ensure good lighting
- Center the main subject
- Avoid heavy compression
