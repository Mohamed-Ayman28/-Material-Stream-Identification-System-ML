"""
Diagnostic script to compare features between static images and camera frames
This helps understand why camera deployment predictions differ from static predictions
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from feature_extraction_enhanced import extract_features
from utils import predict_with_rejection

def analyze_image(image_path, model, scaler, class_map):
    """Analyze static image"""
    print(f"\n{'='*60}")
    print(f"STATIC IMAGE: {image_path}")
    print(f"{'='*60}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load {image_path}")
        return
    
    # Process like predict.py
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (128, 128))
    
    features = extract_features(resized)
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    result = predict_with_rejection(model, features_scaled)
    
    print(f"Image shape: {img.shape}")
    print(f"Resized shape: {resized.shape}")
    print(f"Feature stats - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
    print(f"Feature stats (scaled) - Mean: {features_scaled.mean():.4f}, Std: {features_scaled.std():.4f}")
    print(f"Feature range: [{features.min():.2f}, {features.max():.2f}]")
    print(f"Feature range (scaled): [{features_scaled.min():.2f}, {features_scaled.max():.2f}]")
    
    if result.label_idx is not None:
        class_name = class_map.get(int(result.label_idx), 'Unknown')
        print(f"\n✓ Prediction: {class_name}")
        print(f"  Confidence: {result.confidence:.2%}")
    else:
        print(f"\n✗ Prediction: Unknown ({result.reason})")
        if result.confidence:
            print(f"  Confidence: {result.confidence:.2%}")


def analyze_camera(model, scaler, class_map, num_frames=10):
    """Analyze camera frames"""
    print(f"\n{'='*60}")
    print(f"CAMERA FRAMES (analyzing {num_frames} frames)")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    frame_stats = []
    predictions = []
    
    print("Capturing frames... (press 'q' to stop early)")
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Process like deploy.py
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128))
        
        features = extract_features(resized)
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        result = predict_with_rejection(model, features_scaled)
        
        # Store stats
        frame_stats.append({
            'mean': features.mean(),
            'std': features.std(),
            'min': features.min(),
            'max': features.max(),
            'mean_scaled': features_scaled.mean(),
            'std_scaled': features_scaled.std()
        })
        
        if result.label_idx is not None:
            class_name = class_map.get(int(result.label_idx), 'Unknown')
            predictions.append(f"{class_name} ({result.confidence:.1%})")
        else:
            predictions.append(f"Unknown ({result.reason})")
        
        # Show frame
        cv2.putText(frame, f"Frame {i+1}/{num_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera Diagnostic', frame)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    print(f"\nCaptured {len(frame_stats)} frames")
    print(f"\nFeature Statistics (Camera Frames):")
    print(f"  Mean (raw): {np.mean([s['mean'] for s in frame_stats]):.4f} ± {np.std([s['mean'] for s in frame_stats]):.4f}")
    print(f"  Std (raw):  {np.mean([s['std'] for s in frame_stats]):.4f} ± {np.std([s['std'] for s in frame_stats]):.4f}")
    print(f"  Min (raw):  {np.mean([s['min'] for s in frame_stats]):.4f}")
    print(f"  Max (raw):  {np.mean([s['max'] for s in frame_stats]):.4f}")
    print(f"\n  Mean (scaled): {np.mean([s['mean_scaled'] for s in frame_stats]):.4f} ± {np.std([s['mean_scaled'] for s in frame_stats]):.4f}")
    print(f"  Std (scaled):  {np.mean([s['std_scaled'] for s in frame_stats]):.4f} ± {np.std([s['std_scaled'] for s in frame_stats]):.4f}")
    
    print(f"\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"  Frame {i+1}: {pred}")


def main():
    print("="*60)
    print("CAMERA vs STATIC IMAGE DIAGNOSTIC")
    print("="*60)
    
    # Load model and scaler
    model_path = "models/svm_enhanced.pkl"
    scaler_path = "models/scaler_enhanced_87.pkl"
    
    print(f"\nLoading {model_path}...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load class map
    import json
    with open("models/class_map.json") as f:
        data = json.load(f)
        if isinstance(data, dict) and 'class_map' in data:
            class_map = {int(k): v for k, v in data['class_map'].items()}
        else:
            class_map = {int(k): v for k, v in data.items()}
    
    # Analyze some static images from dataset
    test_images = [
        "dataset/glass/0f5c9c27-9cad-49e5-b6de-6f51723ecb33.jpg",
        "dataset/metal/metal1.jpg" if Path("dataset/metal/metal1.jpg").exists() else None,
        "dataset/plastic/plastic1.jpg" if Path("dataset/plastic/plastic1.jpg").exists() else None,
    ]
    
    # Find actual images
    for material in ['glass', 'metal', 'plastic']:
        folder = Path(f"dataset/{material}")
        if folder.exists():
            images = list(folder.glob("*.jpg"))
            if images:
                test_images.append(str(images[0]))
                break
    
    test_images = [img for img in test_images if img and Path(img).exists()]
    
    for img_path in test_images[:3]:  # Analyze up to 3 images
        analyze_image(img_path, model, scaler, class_map)
    
    # Analyze camera frames
    print("\n" + "="*60)
    input("Press Enter to start camera analysis (or Ctrl+C to skip)...")
    analyze_camera(model, scaler, class_map, num_frames=10)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
