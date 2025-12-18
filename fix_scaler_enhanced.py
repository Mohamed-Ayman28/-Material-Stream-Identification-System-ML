"""
fix_scaler_enhanced.py - Create a scaler that matches the 87-feature enhanced models
"""
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, 'src')

# Load the features that were used for training (features_optimized.npz has 1881, but we need 87-feature version)
# We'll create a dummy scaler fitted to the correct 87-feature space

print("Creating scaler for 87-feature enhanced models...")

# Option 1: Create identity scaler (no scaling)
# This works if models were trained without scaling

# Option 2: Fit scaler to some 87-feature data
# Let's extract 87 features from a few sample images

from feature_extraction_enhanced import extract_features
import cv2
from pathlib import Path

features_list = []
dataset_dir = Path('dataset')

for class_dir in dataset_dir.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.glob('*.jpg'))[:10]  # Sample 10 images per class
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (128, 128))
                feats = extract_features(resized)
                features_list.append(feats)

if features_list:
    X_sample = np.array(features_list)
    print(f"Sample features shape: {X_sample.shape}")
    
    scaler = StandardScaler()
    scaler.fit(X_sample)
    
    joblib.dump(scaler, 'models/scaler_enhanced_87.pkl')
    print("Saved scaler to models/scaler_enhanced_87.pkl")
    print(f"Scaler expects {scaler.n_features_in_} features")
else:
    print("No sample images found!")
