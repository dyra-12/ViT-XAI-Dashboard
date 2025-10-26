# Counterfactual Analysis Test Images

Images for testing prediction robustness through patch perturbations.

## 📸 Recommended Images

### What to Include:
1. **Simple Backgrounds**: Easy to see perturbation effects
2. **Centered Objects**: Better for patch-based analysis
3. **Distinct Features**: Eyes, wheels, wings
4. **Varying Complexity**: From simple to complex

### Current Images:
- `face_portrait.jpg` - Test facial feature importance
- `car_side.jpg` - Test vehicle components (wheels, doors)
- `building.jpg` - Test architectural elements
- `flower.jpg` - Simple object baseline

## 🧪 Testing Guide

### Basic Robustness Test:
```
1. Upload face_portrait.jpg
2. Patch size: 32px
3. Perturbation: blur
4. Check which patches affect prediction most
```

### Feature Importance:
```
1. Upload car_side.jpg
2. Try different perturbation types
3. Identify critical regions (wheels, windows)
```

### Sensitivity Analysis:
```
1. Upload flower.jpg
2. Use blackout perturbation
3. Find minimal critical area
```

## 💡 Tips
- Images with clear, centered subjects work best
- Try all perturbation types (blur, blackout, gray, noise)
- Compare patch sizes (16, 32, 48, 64)
- Look for prediction flip rates
