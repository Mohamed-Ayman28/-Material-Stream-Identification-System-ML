"""
Real-time camera feature normalizer
Apply normalization to camera frames to match training distribution
"""

import cv2
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from feature_extraction_enhanced import extract_features

def collect_camera_samples(num_samples=50):
    """Collect camera frames and extract features"""
    print(f"Collecting {num_samples} camera samples...")
    print("Point camera at various materials in good lighting")
    print("Press 's' to start collection, 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    # Warm up
    for _ in range(30):
        cap.read()
    
    features_list = []
    collecting = False
    
    while len(features_list) < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop center
        h, w = frame.shape[:2]
        crop_h, crop_w = int(h * 0.6), int(w * 0.6)
        start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
        cropped = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
        
        # Show
        cv2.rectangle(frame, (start_x, start_y), 
                     (start_x + crop_w, start_y + crop_h),
                     (0, 255, 0), 2)
        
        status = f"Collected: {len(features_list)}/{num_samples}"
        if not collecting:
            status = "Press 's' to start | " + status
        
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Collect Samples', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            collecting = True
        
        if collecting:
            # Extract features
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (128, 128))
            feat = extract_features(resized)
            features_list.append(feat)
            print(f"Sample {len(features_list)}/{num_samples}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(features_list) == 0:
        return None
    
    return np.array(features_list)


def main():
    print("="*60)
    print("CAMERA FEATURE ANALYSIS & CORRECTION")
    print("="*60)
    
    # Collect camera samples
    camera_features = collect_camera_samples(50)
    
    if camera_features is None:
        print("No samples collected")
        return
    
    print(f"\nCamera features shape: {camera_features.shape}")
    print(f"\nCamera feature statistics (RAW):")
    print(f"  Mean: {camera_features.mean():.4f}")
    print(f"  Std: {camera_features.std():.4f}")
    print(f"  Min: {camera_features.min():.4f}")
    print(f"  Max: {camera_features.max():.4f}")
    
    # Load scaler and check scaled distribution
    scaler = joblib.load("models/scaler_enhanced_87.pkl")
    camera_scaled = scaler.transform(camera_features)
    
    print(f"\nCamera feature statistics (SCALED with training scaler):")
    print(f"  Mean: {camera_scaled.mean():.4f}")
    print(f"  Std: {camera_scaled.std():.4f}")
    print(f"  Min: {camera_scaled.min():.4f}")
    print(f"  Max: {camera_scaled.max():.4f}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")
    print("\nExpected scaled features (from training):")
    print("  Mean: ~0.0")
    print("  Std: ~1.0")
    
    print(f"\nYour camera scaled features:")
    print(f"  Mean: {camera_scaled.mean():.4f}")
    print(f"  Std: {camera_scaled.std():.4f}")
    
    if abs(camera_scaled.mean()) > 0.5 or abs(camera_scaled.std() - 1.0) > 0.5:
        print("\n⚠️  WARNING: Large distribution mismatch!")
        print("Camera features are significantly different from training data.")
        print("\nPossible causes:")
        print("  - Different lighting conditions")
        print("  - Camera auto-exposure/white-balance")
        print("  - Different material backgrounds")
        print("\nRecommendations:")
        print("  1. Retrain with camera-captured images")
        print("  2. Use very low confidence thresholds")
        print("  3. Add data augmentation with varying exposures")
    else:
        print("\n✓ Camera features are reasonably close to training distribution")
        print("The current thresholds should work")
    
    # Save camera feature statistics for reference
    np.savez('camera_feature_stats.npz',
             mean=camera_features.mean(),
             std=camera_features.std(),
             samples=camera_features)
    print("\nSaved camera feature stats to camera_feature_stats.npz")


if __name__ == '__main__':
    main()
