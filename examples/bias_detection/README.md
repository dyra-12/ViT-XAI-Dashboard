# Bias Detection Test Images

Images for testing performance variations across different subgroups.

## 📸 Recommended Images

### What to Include:
1. **Same Subject, Different Conditions**: Day/night, indoor/outdoor
2. **Environmental Variations**: Weather, seasons, lighting
3. **Context Variations**: Urban/rural, natural/artificial
4. **Quality Variations**: Professional vs amateur

### Current Images:
- `dog_daylight.jpg` - Good lighting conditions
- `cat_indoor.jpg` - Controlled indoor environment
- `bird_outdoor.jpg` - Natural outdoor setting
- `urban_scene.jpg` - City environment

## 🧪 Testing Guide

### Lighting Bias:
```
1. Compare dog_daylight.jpg with similar night image
2. Check confidence differences
3. Identify lighting bias if present
```

### Environment Bias:
```
1. Compare cat_indoor.jpg with outdoor cat image
2. Check performance variations
3. Assess environmental impact
```

### Context Bias:
```
1. Use urban_scene.jpg and compare with rural scene
2. Check if model favors certain contexts
3. Review subgroup metrics
```

## 💡 Tips
- Create matched pairs (same subject, different conditions)
- Test systematic variations (brightness, contrast, saturation)
- Document performance differences
- Look for consistent patterns across subgroups
