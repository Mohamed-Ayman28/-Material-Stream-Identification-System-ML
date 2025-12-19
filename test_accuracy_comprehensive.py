"""
Comprehensive accuracy testing on dataset images
"""
import cv2
import numpy as np
import joblib
from pathlib import Path
from src.feature_extraction_enhanced import extract_features
from src.utils import predict_with_rejection

print("="*80)
print("COMPREHENSIVE ACCURACY TEST")
print("="*80)

# Load models and scaler
models = {
    'KNN': joblib.load('models/knn_enhanced.pkl'),
    'SVM': joblib.load('models/svm_enhanced.pkl'),
    'Ensemble': joblib.load('models/ensemble_enhanced.pkl'),
}
scaler = joblib.load('models/scaler_enhanced_87.pkl')

# Class map
class_map = {0: 'glass', 1: 'paper', 2: 'cardboard', 3: 'plastic', 4: 'metal', 5: 'trash'}
reverse_map = {v: k for k, v in class_map.items()}

# Test on dataset images
results = {name: {'correct': 0, 'wrong': 0, 'unknown': 0, 'total': 0} for name in models.keys()}

for class_name in class_map.values():
    class_dir = Path(f'dataset/{class_name}')
    if not class_dir.exists():
        continue
    
    # Get first 20 images from each class
    images = list(class_dir.glob('*.jpg'))[:20]
    
    for img_path in images:
        try:
            # Load and preprocess
            img = cv2.imread(str(img_path))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (128, 128))
            
            # Extract and scale features
            features = extract_features(resized)
            features = features.reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            # Test each model
            for model_name, model in models.items():
                result = predict_with_rejection(model, features_scaled)
                results[model_name]['total'] += 1
                
                if result.label_idx is None:
                    results[model_name]['unknown'] += 1
                else:
                    pred_class = class_map[result.label_idx]
                    if pred_class == class_name:
                        results[model_name]['correct'] += 1
                    else:
                        results[model_name]['wrong'] += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Print results
print(f"\n{'='*80}")
print("ACCURACY RESULTS ON DATASET")
print(f"{'='*80}")
print(f"\n{'Model':<12s} {'Correct':<10s} {'Wrong':<10s} {'Unknown':<10s} {'Total':<10s} {'Accuracy':<10s}")
print("-"*80)

for model_name, stats in results.items():
    accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"{model_name:<12s} {stats['correct']:<10d} {stats['wrong']:<10d} {stats['unknown']:<10d} {stats['total']:<10d} {accuracy:>7.2f}%")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

for model_name, stats in sorted(results.items(), key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0, reverse=True):
    total = stats['total']
    if total > 0:
        accuracy = stats['correct'] / total * 100
        rejection_rate = stats['unknown'] / total * 100
        print(f"\n{model_name}:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Rejection rate: {rejection_rate:.2f}%")
        print(f"  ✓ {'ABOVE 85%' if accuracy >= 85 else 'Below 85% target'}")

print(f"\n{'='*80}")
