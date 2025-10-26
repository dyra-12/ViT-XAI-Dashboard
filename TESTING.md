# 🧪 Testing Guide for ViT Auditing Toolkit

Complete guide for testing all features using the provided sample images.

## 📋 Quick Test Checklist

- [ ] Basic Explainability - Attention Visualization
- [ ] Basic Explainability - GradCAM
- [ ] Basic Explainability - GradientSHAP
- [ ] Counterfactual Analysis - All perturbation types
- [ ] Confidence Calibration - Different bin sizes
- [ ] Bias Detection - Multiple subgroups
- [ ] Model Switching (ViT-Base ↔ ViT-Large)

---

## 🔍 Tab 1: Basic Explainability Testing

### Test 1: Attention Visualization
**Image**: `examples/basic_explainability/cat_portrait.jpg`

**Steps**:
1. Load ViT-Base model
2. Upload cat_portrait.jpg
3. Select "Attention Visualization"
4. Try these layer/head combinations:
   - Layer 0, Head 0 (low-level features)
   - Layer 6, Head 0 (mid-level patterns)
   - Layer 11, Head 0 (high-level semantics)

**Expected Results**:
- ✅ Early layers: Focus on edges, textures
- ✅ Middle layers: Focus on cat features (ears, eyes)
- ✅ Late layers: Focus on discriminative regions (face)

---

### Test 2: GradCAM Visualization
**Image**: `examples/basic_explainability/sports_car.jpg`

**Steps**:
1. Upload sports_car.jpg
2. Select "GradCAM" method
3. Click "Analyze Image"

**Expected Results**:
- ✅ Heatmap highlights car body, wheels
- ✅ Prediction confidence > 70%
- ✅ Top class includes "sports car" or "convertible"

---

### Test 3: GradientSHAP
**Image**: `examples/basic_explainability/bird_flying.jpg`

**Steps**:
1. Upload bird_flying.jpg
2. Select "GradientSHAP" method
3. Wait for analysis (takes ~10-15 seconds)

**Expected Results**:
- ✅ Attribution map shows bird outline
- ✅ Wings and body highlighted
- ✅ Background has low attribution

---

### Test 4: Multiple Objects
**Image**: `examples/basic_explainability/coffee_cup.jpg`

**Steps**:
1. Upload coffee_cup.jpg
2. Try all three methods
3. Compare explanations

**Expected Results**:
- ✅ All methods highlight the cup
- ✅ Consistent predictions across methods
- ✅ Some variation in exact highlighted regions

---

## 🔄 Tab 2: Counterfactual Analysis Testing

### Test 5: Face Feature Importance
**Image**: `examples/counterfactual/face_portrait.jpg`

**Steps**:
1. Upload face_portrait.jpg
2. Settings:
   - Patch size: 32
   - Perturbation: blur
3. Click "Run Counterfactual Analysis"

**Expected Results**:
- ✅ Face region shows high sensitivity
- ✅ Background regions have low impact
- ✅ Prediction flip rate < 50%

---

### Test 6: Vehicle Components
**Image**: `examples/counterfactual/car_side.jpg`

**Steps**:
1. Upload car_side.jpg
2. Test each perturbation type:
   - Blur
   - Blackout
   - Gray
   - Noise
3. Compare results

**Expected Results**:
- ✅ Wheels are critical regions
- ✅ Windows/doors moderately important
- ✅ Blackout causes most disruption

---

### Test 7: Architectural Elements
**Image**: `examples/counterfactual/building.jpg`

**Steps**:
1. Upload building.jpg
2. Patch size: 48
3. Perturbation: gray

**Expected Results**:
- ✅ Structural elements highlighted
- ✅ Lower flip rate (buildings are robust)
- ✅ Consistent confidence across patches

---

### Test 8: Simple Object Baseline
**Image**: `examples/counterfactual/flower.jpg`

**Steps**:
1. Upload flower.jpg
2. Try smallest patch size (16)
3. Use blackout perturbation

**Expected Results**:
- ✅ Flower center most critical
- ✅ Petals moderately important
- ✅ Background has minimal impact

---

## 📊 Tab 3: Confidence Calibration Testing

### Test 9: High-Quality Image
**Image**: `examples/calibration/clear_panda.jpg`

**Steps**:
1. Upload clear_panda.jpg
2. Number of bins: 10
3. Run analysis

**Expected Results**:
- ✅ High mean confidence (> 0.8)
- ✅ Low overconfident rate
- ✅ Calibration curve near diagonal

---

### Test 10: Complex Scene
**Image**: `examples/calibration/workspace.jpg`

**Steps**:
1. Upload workspace.jpg
2. Number of bins: 15
3. Compare with panda results

**Expected Results**:
- ✅ Lower mean confidence (multiple objects)
- ✅ Higher variance in predictions
- ✅ More distributed across bins

---

### Test 11: Bin Size Comparison
**Image**: `examples/calibration/outdoor_scene.jpg`

**Steps**:
1. Upload outdoor_scene.jpg
2. Test with bins: 5, 10, 20
3. Compare calibration curves

**Expected Results**:
- ✅ More bins = finer granularity
- ✅ General trend consistent
- ✅ 10 bins usually optimal

---

## ⚖️ Tab 4: Bias Detection Testing

### Test 12: Lighting Conditions
**Image**: `examples/bias_detection/dog_daylight.jpg`

**Steps**:
1. Upload dog_daylight.jpg
2. Run bias detection
3. Note confidence for daylight subgroup

**Expected Results**:
- ✅ 4 subgroups generated (original, bright+, bright-, contrast+)
- ✅ Confidence varies across subgroups
- ✅ Original has highest confidence typically

---

### Test 13: Indoor vs Outdoor
**Images**: 
- `examples/bias_detection/cat_indoor.jpg`
- `examples/bias_detection/bird_outdoor.jpg`

**Steps**:
1. Test both images separately
2. Compare confidence distributions
3. Note any systematic differences

**Expected Results**:
- ✅ Both should predict correctly
- ✅ Confidence may vary
- ✅ Subgroup metrics show variations

---

### Test 14: Urban Environment
**Image**: `examples/bias_detection/urban_scene.jpg`

**Steps**:
1. Upload urban_scene.jpg
2. Run bias detection
3. Check for environmental bias

**Expected Results**:
- ✅ Multiple objects detected
- ✅ Varied confidence across subgroups
- ✅ Brightness variations affect predictions

---

## 🎯 Cross-Tab Testing

### Test 15: Same Image, All Tabs
**Image**: `examples/general/pizza.jpg`

**Steps**:
1. Tab 1: Check predictions and explanations
2. Tab 2: Test robustness with perturbations
3. Tab 3: Check confidence calibration
4. Tab 4: Analyze across subgroups

**Expected Results**:
- ✅ Consistent predictions across tabs
- ✅ High confidence (pizza is clear class)
- ✅ Robust to perturbations
- ✅ Well-calibrated

---

### Test 16: Model Comparison
**Image**: `examples/general/laptop.jpg`

**Steps**:
1. Load ViT-Base, analyze laptop.jpg in Tab 1
2. Note top predictions and confidence
3. Load ViT-Large, analyze same image
4. Compare results

**Expected Results**:
- ✅ ViT-Large slightly higher confidence
- ✅ Similar top predictions
- ✅ Better attention patterns (Large)
- ✅ Longer inference time (Large)

---

### Test 17: Edge Case Testing
**Image**: `examples/general/mountain.jpg`

**Steps**:
1. Test in all tabs
2. Note predictions (landscape/nature)
3. Check explanation quality

**Expected Results**:
- ✅ May predict multiple classes (mountain, valley, landscape)
- ✅ Lower confidence (ambiguous category)
- ✅ Attention spread across scene

---

### Test 18: Furniture Classification
**Image**: `examples/general/chair.jpg`

**Steps**:
1. Basic explainability test
2. Counterfactual with blur
3. Check which parts are critical

**Expected Results**:
- ✅ Predicts chair/furniture
- ✅ Legs and seat are critical
- ✅ Background less important

---

## 🔧 Performance Testing

### Test 19: Load Time
**Steps**:
1. Clear browser cache
2. Time model loading
3. Note first analysis time vs subsequent

**Expected**:
- First load: 5-15 seconds
- Subsequent: < 1 second
- Analysis: 2-5 seconds per image

---

### Test 20: Memory Usage
**Steps**:
1. Open browser dev tools
2. Monitor memory during analysis
3. Test with both models

**Expected**:
- ViT-Base: ~2GB RAM
- ViT-Large: ~4GB RAM
- No memory leaks over multiple analyses

---

## 🐛 Error Handling Testing

### Test 21: Invalid Inputs
**Steps**:
1. Try uploading non-image file
2. Try very large image (> 50MB)
3. Try corrupted image

**Expected**:
- ✅ Graceful error messages
- ✅ No crashes
- ✅ User-friendly feedback

---

### Test 22: Edge Cases
**Steps**:
1. Try extremely dark/bright images
2. Try pure noise images
3. Try text-only images

**Expected**:
- ✅ Model makes predictions
- ✅ Lower confidence expected
- ✅ Explanations still generated

---

## 📝 Test Results Template

```markdown
## Test Session: [Date]

**Tester**: [Name]
**Model**: ViT-Base / ViT-Large
**Browser**: [Chrome/Firefox/Safari]
**Environment**: [Local/Docker/Cloud]

### Results Summary:
- Tests Passed: __/22
- Tests Failed: __/22
- Critical Issues: __
- Minor Issues: __

### Detailed Results:

#### Test 1: Attention Visualization
- Status: ✅ Pass / ❌ Fail
- Notes: [observations]

[Continue for all tests...]

### Issues Found:
1. [Issue description]
   - Severity: Critical/Major/Minor
   - Steps to reproduce:
   - Expected: 
   - Actual:

### Recommendations:
- [Improvement suggestions]
```

---

## 🚀 Quick Smoke Test (5 minutes)

Fastest way to verify everything works:

```bash
# 1. Start app
python app.py

# 2. Load ViT-Base model

# 3. Quick tests:
Tab 1: Upload examples/basic_explainability/cat_portrait.jpg → Analyze
Tab 2: Upload examples/counterfactual/flower.jpg → Analyze
Tab 3: Upload examples/calibration/clear_panda.jpg → Analyze
Tab 4: Upload examples/bias_detection/dog_daylight.jpg → Analyze

# 4. All should complete without errors
```

---

## 📊 Automated Testing

Run automated tests:

```bash
# Unit tests
pytest tests/test_phase1_complete.py -v

# Advanced features tests
pytest tests/test_advanced_features.py -v

# All tests with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🎓 User Acceptance Testing

**Scenario 1: First-time User**
- Can they understand the interface?
- Can they complete basic analysis?
- Is documentation helpful?

**Scenario 2: Researcher**
- Can they compare multiple methods?
- Can they export results?
- Is explanation quality sufficient?

**Scenario 3: ML Practitioner**
- Can they validate their model?
- Are metrics meaningful?
- Can they identify issues?

---

## ✅ Sign-off Criteria

Before considering testing complete:

- [ ] All 22 tests pass
- [ ] No critical bugs
- [ ] Performance acceptable
- [ ] Documentation accurate
- [ ] User feedback positive
- [ ] All tabs functional
- [ ] Both models work
- [ ] Error handling robust

---

**Happy Testing! 🎉**

For issues or questions, see [CONTRIBUTING.md](CONTRIBUTING.md)
