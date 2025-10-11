# test_phase1_complete.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_loader import load_model_and_processor, SUPPORTED_MODELS
from predictor import predict_image, create_prediction_plot
from explainer import explain_attention, explain_gradcam, explain_gradient_shap
from utils import preprocess_image, create_comparison_figure, get_top_predictions_dict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def test_phase1_complete():
    """
    Complete Phase 1 Test - Tests all components together.
    """
    print("🧪 ViT Auditing Toolkit - Phase 1 Complete Test")
    print("=" * 50)
    
    try:
        # Test 1: Model Loading
        print("1. Testing Model Loading...")
        model, processor = load_model_and_processor()
        print(f"   ✅ Loaded: {SUPPORTED_MODELS['ViT-Base']}")
        
        # Test 2: Create test image using utils
        print("2. Testing Image Preprocessing...")
        # Create a more realistic test image
        test_image = Image.new('RGB', (300, 200), color=(150, 75, 75))
        # Add different colored regions
        for x in range(50, 150):
            for y in range(50, 150):
                test_image.putpixel((x, y), (75, 150, 75))  # Green rectangle
        for x in range(180, 280):
            for y in range(30, 100):
                test_image.putpixel((x, y), (75, 75, 150))  # Blue rectangle
        
        # Preprocess using utils
        processed_image = preprocess_image(test_image, target_size=224)
        print(f"   ✅ Original size: {test_image.size}, Processed: {processed_image.size}")
        
        # Test 3: Prediction Pipeline
        print("3. Testing Prediction Pipeline...")
        probs, indices, labels = predict_image(processed_image, model, processor, top_k=5)
        pred_fig = create_prediction_plot(probs, labels)
        
        # Test utils function
        pred_dict = get_top_predictions_dict(probs, labels)
        print(f"   ✅ Top prediction: {labels[0]} ({probs[0]:.2%})")
        
        # Test 4: Attention Explanation
        print("4. Testing Attention Visualization...")
        attention_fig = explain_attention(model, processor, processed_image, layer_index=6, head_index=0)
        print("   ✅ Attention visualization generated")
        
        # Test 5: GradCAM Explanation
        print("5. Testing GradCAM...")
        gradcam_fig, gradcam_overlay = explain_gradcam(model, processor, processed_image)
        print("   ✅ GradCAM visualization generated")
        
        # Test 6: GradientSHAP Explanation
        print("6. Testing GradientSHAP...")
        shap_fig = explain_gradient_shap(model, processor, processed_image, n_samples=3)
        print("   ✅ GradientSHAP visualization generated")
        
        # Test 7: Utils - Comparison Figure
        print("7. Testing Utils - Comparison Figure...")
        comparison_fig = create_comparison_figure(
            processed_image,
            [gradcam_overlay],
            ['GradCAM Overlay']
        )
        print("   ✅ Comparison figure generated")
        
        # Display Results
        print("\n📊 DISPLAYING RESULTS:")
        print("=" * 30)
        
        # Show prediction results
        plt.figure(pred_fig.number)
        plt.suptitle("1. Model Predictions", fontweight='bold', y=1.02)
        plt.show()
        
        # Show attention results
        plt.figure(attention_fig.number)
        plt.suptitle("2. Attention Visualization", fontweight='bold', y=1.02)
        plt.show()
        
        # Show GradCAM results
        plt.figure(gradcam_fig.number)
        plt.suptitle("3. GradCAM Explanation", fontweight='bold', y=1.02)
        plt.show()
        
        # Show SHAP results
        plt.figure(shap_fig.number)
        plt.suptitle("4. GradientSHAP Explanation", fontweight='bold', y=1.02)
        plt.show()
        
        # Show comparison
        plt.figure(comparison_fig.number)
        plt.suptitle("5. Comparison View", fontweight='bold', y=1.02)
        plt.show()
        
        # Summary
        print("\n🎉 PHASE 1 COMPLETE SUMMARY:")
        print("=" * 35)
        print("✅ Model Loading & Preprocessing")
        print("✅ Prediction Pipeline with Visualization") 
        print("✅ Attention Visualization")
        print("✅ GradCAM Explanations")
        print("✅ GradientSHAP Explanations")
        print("✅ Utility Functions")
        print(f"✅ All components integrated successfully!")
        print("\n🚀 Ready for Phase 2: Dashboard Integration!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 1 Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """
    Test individual components for debugging.
    """
    print("\n🔧 Individual Component Tests:")
    print("-" * 30)
    
    try:
        # Test model loading
        model, processor = load_model_and_processor()
        print("✅ Model loading: PASS")
        
        # Test image creation
        test_img = Image.new('RGB', (224, 224), color='red')
        print("✅ Image creation: PASS")
        
        # Test prediction
        probs, indices, labels = predict_image(test_img, model, processor)
        print("✅ Prediction: PASS")
        
        # Test attention
        attn_fig = explain_attention(model, processor, test_img)
        print("✅ Attention: PASS")
        
        # Test GradCAM
        gc_fig, gc_img = explain_gradcam(model, processor, test_img)
        print("✅ GradCAM: PASS")
        
        # Test SHAP
        shap_fig = explain_gradient_shap(model, processor, test_img, n_samples=2)
        print("✅ GradientSHAP: PASS")
        
        # Test utils
        from utils import normalize_heatmap
        test_heatmap = np.random.rand(10, 10)
        normalized = normalize_heatmap(test_heatmap)
        print("✅ Utils: PASS")
        
        print("\n🎉 All individual components working!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")

if __name__ == "__main__":
    # Run complete test
    success = test_phase1_complete()
    
    if success:
        # Run quick individual tests
        test_individual_components()
    else:
        print("\n⚠️  Running individual component tests for debugging...")
        test_individual_components()