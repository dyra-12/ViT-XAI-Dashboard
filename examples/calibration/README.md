# Confidence Calibration Test Images

Images with varying quality levels to test confidence calibration.

## 📸 Recommended Images

### What to Include:
1. **High Quality**: Clear, well-lit images (should have high confidence)
2. **Medium Quality**: Slightly challenging images
3. **Low Quality**: Blurry, dark, or pixelated
4. **Edge Cases**: Partial objects, occlusions

### Current Images:
- `clear_panda.jpg` - High quality, should be confident
- `outdoor_scene.jpg` - Medium difficulty
- `workspace.jpg` - Complex scene with multiple objects

## 🧪 Testing Guide

### Calibration Baseline:
```
1. Upload clear_panda.jpg
2. Note confidence level (should be high)
3. Check if it matches prediction accuracy
```

### Quality Impact:
```
1. Test with images of different quality
2. Observe confidence changes
3. Check calibration curve alignment
```

### Bin Analysis:
```
1. Try different bin counts (5, 10, 20)
2. See how granularity affects calibration
3. Identify overconfident regions
```

## 💡 Tips
- Include images you know the correct label for
- Mix easy and hard examples
- Test with various lighting conditions
- Compare confidence across similar images
