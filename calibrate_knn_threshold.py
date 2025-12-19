"""
Calibrate KNN distance thresholds by testing on training data
"""
import numpy as np
import joblib
import cv2
from pathlib import Path
from src.feature_extraction_enhanced import extract_features

# Load model and scaler
knn_model = joblib.load('models/knn_enhanced.pkl')
scaler = joblib.load('models/scaler_enhanced_87.pkl')

print("="*80)
print("KNN DISTANCE CALIBRATION")
print("="*80)

# Test on dataset images (known classes)
test_images = []
for class_dir in ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']:
    class_path = Path(f'dataset/{class_dir}')
    if class_path.exists():
        # Get first 5 images from each class
        images = list(class_path.glob('*.jpg'))[:5]
        for img_path in images:
            test_images.append((str(img_path), class_dir))

print(f"\nTesting {len(test_images)} dataset images...")

distances_all = []
distances_correct = []
distances_wrong = []

for img_path, true_class in test_images:
    try:
        img = cv2.imread(img_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128))
        
        # Extract and scale features
        features = extract_features(resized)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Get distances
        distances, neighbors = knn_model.kneighbors(features_scaled)
        mean_dist = np.mean(distances)
        
        # Get prediction
        pred_idx = knn_model.predict(features_scaled)[0]
        
        # Class mapping
        class_map = {0: 'glass', 1: 'paper', 2: 'cardboard', 3: 'plastic', 4: 'metal', 5: 'trash'}
        pred_class = class_map[pred_idx]
        
        distances_all.append(mean_dist)
        if pred_class == true_class:
            distances_correct.append(mean_dist)
        else:
            distances_wrong.append(mean_dist)
            
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print(f"\n{'='*80}")
print("DISTANCE STATISTICS")
print(f"{'='*80}")

print(f"\nAll predictions ({len(distances_all)} samples):")
print(f"  Min distance:  {np.min(distances_all):.4f}")
print(f"  Max distance:  {np.max(distances_all):.4f}")
print(f"  Mean distance: {np.mean(distances_all):.4f}")
print(f"  Median distance: {np.median(distances_all):.4f}")
print(f"  Std distance:  {np.std(distances_all):.4f}")
print(f"  25th percentile: {np.percentile(distances_all, 25):.4f}")
print(f"  75th percentile: {np.percentile(distances_all, 75):.4f}")
print(f"  95th percentile: {np.percentile(distances_all, 95):.4f}")

print(f"\nCorrect predictions ({len(distances_correct)} samples):")
print(f"  Min distance:  {np.min(distances_correct):.4f}")
print(f"  Max distance:  {np.max(distances_correct):.4f}")
print(f"  Mean distance: {np.mean(distances_correct):.4f}")
print(f"  Median distance: {np.median(distances_correct):.4f}")
print(f"  95th percentile: {np.percentile(distances_correct, 95):.4f}")

if distances_wrong:
    print(f"\nWrong predictions ({len(distances_wrong)} samples):")
    print(f"  Min distance:  {np.min(distances_wrong):.4f}")
    print(f"  Max distance:  {np.max(distances_wrong):.4f}")
    print(f"  Mean distance: {np.mean(distances_wrong):.4f}")

print(f"\n{'='*80}")
print("THRESHOLD RECOMMENDATIONS")
print(f"{'='*80}")

# Conservative: accept 95% of correct predictions
threshold_95 = np.percentile(distances_correct, 95)
print(f"\nConservative (95th percentile of correct): {threshold_95:.2f}")
print(f"  → Accepts 95% of known good predictions")

# Moderate: accept 99% of correct predictions
threshold_99 = np.percentile(distances_correct, 99) if len(distances_correct) > 10 else threshold_95
print(f"\nModerate (99th percentile of correct): {threshold_99:.2f}")
print(f"  → Accepts 99% of known good predictions")

# Permissive: mean + 2*std
threshold_mean_std = np.mean(distances_correct) + 2 * np.std(distances_correct)
print(f"\nPermissive (mean + 2*std): {threshold_mean_std:.2f}")
print(f"  → Statistical approach")

# Current threshold
print(f"\nCurrent threshold: 0.50")
print(f"  → Accepts {np.sum(np.array(distances_correct) <= 0.50)} / {len(distances_correct)} correct predictions ({100*np.sum(np.array(distances_correct) <= 0.50)/len(distances_correct):.1f}%)")

print(f"\n{'='*80}")
print("RECOMMENDATION:")
print(f"  Use threshold: {max(threshold_99, threshold_95):.2f}")
print(f"{'='*80}")
