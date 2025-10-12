# test_advanced_features.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_loader import load_model_and_processor
from auditor import create_auditors, CounterfactualAnalyzer, ConfidenceCalibrationAnalyzer, BiasDetector
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def create_test_subsets():
    """Create dummy test subsets for bias detection demo."""
    # Create different colored images to simulate subgroups
    subsets = []
    subset_names = ['Red Dominant', 'Green Dominant', 'Blue Dominant', 'Mixed Colors']
    
    for i, name in enumerate(subset_names):
        subset = []
        for j in range(10):  # 10 images per subset
            if name == 'Red Dominant':
                img = Image.new('RGB', (224, 224), color=(200, 50, 50))
            elif name == 'Green Dominant':
                img = Image.new('RGB', (224, 224), color=(50, 200, 50))
            elif name == 'Blue Dominant':
                img = Image.new('RGB', (224, 224), color=(50, 50, 200))
            else:  # Mixed
                color = (50 + j*20, 100 + j*10, 150 - j*15)
                img = Image.new('RGB', (224, 224), color=color)
            subset.append(img)
        subsets.append(subset)
    
    return subsets, subset_names

def test_advanced_features():
    """
    Test the advanced auditing features.
    """
    print("🔬 Testing Advanced Auditing Features")
    print("=" * 50)
    
    try:
        # Load model
        model, processor = load_model_and_processor()
        
        # Create auditors
        auditors = create_auditors(model, processor)
        print("✅ Auditors created: Counterfactual, Calibration, Bias Detection")
        
        # Create test image
        test_image = Image.new('RGB', (224, 224), color=(150, 100, 100))
        for x in range(50, 150):
            for y in range(50, 150):
                test_image.putpixel((x, y), (100, 200, 100))
        
        print("\n1. Testing Counterfactual Analysis...")
        counterfactual_results = auditors['counterfactual'].patch_perturbation_analysis(
            test_image, patch_size=32, perturbation_type='blur'
        )
        print("   ✅ Counterfactual analysis completed")
        print(f"   📊 Avg confidence change: {counterfactual_results['avg_confidence_change']:.4f}")
        print(f"   🔀 Prediction flip rate: {counterfactual_results['prediction_flip_rate']:.2%}")
        
        print("\n2. Testing Confidence Calibration...")
        # Create dummy test set
        test_images = [test_image] * 5  # Simple test with same image
        calibration_results = auditors['calibration'].analyze_calibration(test_images)
        print("   ✅ Calibration analysis completed")
        print(f"   📈 Mean confidence: {calibration_results['metrics']['mean_confidence']:.3f}")
        print(f"   🎯 Overconfident rate: {calibration_results['metrics']['overconfident_rate']:.2%}")
        
        print("\n3. Testing Bias Detection...")
        test_subsets, subset_names = create_test_subsets()
        bias_results = auditors['bias'].analyze_subgroup_performance(test_subsets, subset_names)
        print("   ✅ Bias detection analysis completed")
        print(f"   📊 Analyzed {len(subset_names)} subgroups")
        
        # Display results
        print("\n📊 DISPLAYING ADVANCED ANALYSIS RESULTS:")
        print("=" * 40)
        
        # Counterfactual results
        plt.figure(counterfactual_results['figure'].number)
        plt.suptitle("1. Counterfactual Analysis - Patch Sensitivity", fontweight='bold', y=0.98)
        plt.show()
        
        # Calibration results
        plt.figure(calibration_results['figure'].number)
        plt.suptitle("2. Confidence Calibration Analysis", fontweight='bold', y=0.98)
        plt.show()
        
        # Bias detection results
        plt.figure(bias_results['figure'].number)
        plt.suptitle("3. Bias Detection - Subgroup Analysis", fontweight='bold', y=0.98)
        plt.show()
        
        # Print detailed metrics
        print("\n📈 DETAILED METRICS:")
        print("-" * 20)
        
        print("\n🎯 Counterfactual Analysis:")
        for key, value in counterfactual_results.items():
            if key != 'figure':
                print(f"   {key}: {value}")
        
        print("\n📊 Calibration Analysis:")
        for key, value in calibration_results['metrics'].items():
            print(f"   {key}: {value}")
        
        print("\n⚖️ Bias Detection:")
        print("   Subgroup Metrics:")
        for subgroup, metrics in bias_results['subgroup_metrics'].items():
            print(f"     {subgroup}:")
            for metric, value in metrics.items():
                print(f"       {metric}: {value}")
        
        print("\n🎉 ADVANCED FEATURES SUMMARY:")
        print("=" * 35)
        print("✅ Counterfactual Analysis - Patch Sensitivity")
        print("✅ Confidence Calibration - Reliability Analysis") 
        print("✅ Bias Detection - Subgroup Performance")
        print("✅ All advanced auditing features working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_advanced_features()
    
    if success:
        print("\n🚀 All Phase 1 + Advanced Features Complete!")
        print("   Ready for Phase 2: Dashboard Integration!")
    else:
        print("\n⚠️ Some advanced features need debugging")