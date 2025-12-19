"""
Diagnostic script to understand why KNN predicts everything as Unknown
"""
import numpy as np
import joblib
import cv2
from pathlib import Path
from src.feature_extraction import extract_features
from src.utils import predict_with_rejection

# Load models and scalers
print("="*60)
print("DIAGNOSING KNN BEHAVIOR")
print("="*60)

# Load KNN model
knn_model = joblib.load('models/knn_enhanced.pkl')
scaler = joblib.load('models/scaler_enhanced_87.pkl')  # Use matching scaler!

print(f"\nKNN Model Parameters:")
print(f"  n_neighbors: {knn_model.n_neighbors}")
print(f"  weights: {knn_model.weights}")
print(f"  metric: {knn_model.metric}")
print(f"  algorithm: {knn_model.algorithm}")

print(f"\nScaler expected features: {scaler.n_features_in_}")

# Test with a dataset image
test_image_path = "dataset/cardboard/00ae0969-e9f4-45f4-bfd1-0a0e9bce41d8.jpg"
if Path(test_image_path).exists():
    img = cv2.imread(test_image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (128, 128))
    
    # Extract features using the CORRECT feature extractor (enhanced = 87 features)
    from src.feature_extraction_enhanced import extract_features
    features = extract_features(resized)
    print(f"\nExtracted features: {len(features)}")
    features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Get KNN distances and neighbors
    distances, neighbors = knn_model.kneighbors(features_scaled, n_neighbors=knn_model.n_neighbors)
    mean_distance = np.mean(distances)
    
    print(f"\nKNN Neighbor Analysis:")
    print(f"  Neighbor distances: {distances[0]}")
    print(f"  Mean distance: {mean_distance:.4f}")
    print(f"  Neighbor labels: {knn_model.predict(knn_model._fit_X[neighbors[0]])}")
    
    # Get prediction
    pred = knn_model.predict(features_scaled)[0]
    print(f"  Raw prediction: {pred}")
    
    # Test rejection logic with different thresholds
    print(f"\nRejection Logic Test:")
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        result = predict_with_rejection(knn_model, features_scaled, knn_max_mean_distance=threshold)
        status = "ACCEPTED" if result.label_idx is not None else "REJECTED"
        print(f"  Threshold {threshold:.2f}: {status} (conf={result.confidence:.4f}, reason={result.reason})")
        if result.extra:
            print(f"    Extra: {result.extra}")
    
    # Compare with scaled training data stats
    print(f"\nScaled Feature Statistics:")
    print(f"  Feature mean: {np.mean(features_scaled):.4f}")
    print(f"  Feature std: {np.std(features_scaled):.4f}")
    print(f"  Feature min: {np.min(features_scaled):.4f}")
    print(f"  Feature max: {np.max(features_scaled):.4f}")
    
    # Check if model has predict_proba
    if hasattr(knn_model, 'predict_proba'):
        print(f"\nKNN has predict_proba capability")
        probs = knn_model.predict_proba(features_scaled)[0]
        print(f"  Probabilities: {probs}")
        print(f"  Max prob: {np.max(probs):.4f}")
    else:
        print(f"\nKNN does NOT have predict_proba")

else:
    print(f"Test image not found: {test_image_path}")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
