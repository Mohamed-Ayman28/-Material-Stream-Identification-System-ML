"""
Test camera prediction accuracy by comparing with manual labels
This helps calibrate thresholds specifically for camera deployment
"""

import cv2
import numpy as np
import joblib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from feature_extraction_enhanced import extract_features
from utils import predict_with_rejection

def main():
    print("="*60)
    print("CAMERA ACCURACY TEST")
    print("="*60)
    print("\nInstructions:")
    print("1. Hold each material type in the detection zone")
    print("2. Press key for what you're showing:")
    print("   g = glass, p = plastic, m = metal")
    print("   c = cardboard, a = paper, t = trash")
    print("3. System will show if it predicted correctly")
    print("4. Press 'q' to quit and see statistics")
    print("="*60)
    
    # Load model
    model_path = "models/svm_enhanced.pkl"
    scaler_path = "models/scaler_enhanced_87.pkl"
    class_map_path = "models/class_map.json"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(class_map_path) as f:
        data = json.load(f)
        if isinstance(data, dict) and 'class_map' in data:
            class_map = {int(k): v for k, v in data['class_map'].items()}
        else:
            class_map = {int(k): v for k, v in data.items()}
    
    # Reverse map for user input
    name_to_idx = {v: int(k) for k, v in class_map.items()}
    
    # Key mappings
    key_map = {
        ord('g'): 'glass',
        ord('p'): 'plastic',
        ord('m'): 'metal',
        ord('c'): 'cardboard',
        ord('a'): 'paper',
        ord('t'): 'trash'
    }
    
    # Statistics
    stats = {
        'correct': 0,
        'wrong': 0,
        'unknown': 0,
        'by_class': {}
    }
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    # Warm up
    print("\nWarming up camera...")
    for _ in range(30):
        cap.read()
    print("Ready!")
    
    window_name = 'Camera Test - Press material key when showing object'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    current_prediction = None
    current_confidence = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop center region
            h, w = frame.shape[:2]
            crop_h, crop_w = int(h * 0.6), int(w * 0.6)
            start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
            cropped = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
            
            # Process
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (128, 128))
            
            feat = extract_features(resized)
            feat = scaler.transform(feat.reshape(1, -1))
            
            result = predict_with_rejection(model, feat)
            
            if result.label_idx is not None:
                current_prediction = class_map[int(result.label_idx)]
                current_confidence = result.confidence
            else:
                current_prediction = "Unknown"
                current_confidence = result.confidence if result.confidence else 0.0
            
            # Draw
            cv2.rectangle(frame, (start_x, start_y), 
                         (start_x + crop_w, start_y + crop_h),
                         (0, 255, 0), 2)
            
            # Show prediction
            pred_text = f"Prediction: {current_prediction} ({current_confidence:.1%})"
            cv2.putText(frame, pred_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(frame, "Press material key: g/p/m/c/a/t | q=quit", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show stats
            total = stats['correct'] + stats['wrong'] + stats['unknown']
            if total > 0:
                acc = stats['correct'] / total * 100
                cv2.putText(frame, f"Tests: {total} | Correct: {stats['correct']} ({acc:.1f}%)", 
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key in key_map:
                # User labeled the material
                true_material = key_map[key]
                
                if current_prediction == "Unknown":
                    stats['unknown'] += 1
                    result_text = f"UNKNOWN (should be {true_material})"
                    color = (0, 165, 255)  # Orange
                elif current_prediction.lower() == true_material:
                    stats['correct'] += 1
                    result_text = f"✓ CORRECT ({true_material})"
                    color = (0, 255, 0)  # Green
                else:
                    stats['wrong'] += 1
                    result_text = f"✗ WRONG (predicted {current_prediction}, was {true_material})"
                    color = (0, 0, 255)  # Red
                
                # Track by class
                if true_material not in stats['by_class']:
                    stats['by_class'][true_material] = {'correct': 0, 'total': 0}
                stats['by_class'][true_material]['total'] += 1
                if current_prediction.lower() == true_material:
                    stats['by_class'][true_material]['correct'] += 1
                
                print(f"{result_text} (confidence: {current_confidence:.1%})")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    total = stats['correct'] + stats['wrong'] + stats['unknown']
    if total > 0:
        print(f"\nTotal tests: {total}")
        print(f"Correct: {stats['correct']} ({stats['correct']/total*100:.1f}%)")
        print(f"Wrong: {stats['wrong']} ({stats['wrong']/total*100:.1f}%)")
        print(f"Unknown: {stats['unknown']} ({stats['unknown']/total*100:.1f}%)")
        
        print(f"\nPer-class accuracy:")
        for material, data in sorted(stats['by_class'].items()):
            acc = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
            print(f"  {material:12s}: {data['correct']}/{data['total']} ({acc:.1f}%)")
    else:
        print("\nNo tests performed")

if __name__ == '__main__':
    main()
