"""
Test all images in my_images folder and show rejection behavior
"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
import joblib
import cv2
import numpy as np
from feature_extraction_enhanced import extract_features
from utils import predict_with_rejection
import json

# Load model, scaler, class_map
model = joblib.load('models/ensemble_enhanced.pkl')
scaler = joblib.load('models/scaler_enhanced_87.pkl')

with open('models/class_map_enhanced.json', 'r') as f:
    class_map = {int(k): v for k, v in json.load(f).items()}

# Test all images
my_images = Path('my_images')
results = []

print("\n" + "="*80)
print("TESTING REJECTION MECHANISM ON ALL IMAGES")
print("="*80 + "\n")

for img_path in sorted(my_images.glob('*.*')):
    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue
    
    # Load and process image
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (128, 128))
    
    # Extract and scale features
    features = extract_features(resized).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Predict with rejection
    result = predict_with_rejection(model, features_scaled, model_kind='ensemble')
    
    # Get class name
    if result.label_idx is not None:
        class_name = class_map.get(result.label_idx, f'Class_{result.label_idx}')
        status = '✓ ACCEPTED'
    else:
        class_name = 'Unknown'
        status = '✗ REJECTED'
    
    # Display result
    conf_str = f'{result.confidence*100:.2f}%' if result.confidence else 'N/A'
    print(f"{img_path.name:30s} -> {class_name:12s} ({conf_str:7s}) {status:12s} [{result.reason}]")
    
    results.append({
        'image': img_path.name,
        'prediction': class_name,
        'confidence': result.confidence,
        'accepted': result.label_idx is not None,
        'reason': result.reason
    })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
accepted = sum(1 for r in results if r['accepted'])
rejected = len(results) - accepted
print(f"Total images: {len(results)}")
print(f"Accepted: {accepted} ({accepted/len(results)*100:.1f}%)")
print(f"Rejected (Unknown): {rejected} ({rejected/len(results)*100:.1f}%)")
print("\nRejected images:")
for r in results:
    if not r['accepted']:
        print(f"  - {r['image']}: {r['reason']}")
