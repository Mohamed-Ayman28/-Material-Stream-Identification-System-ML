"""
predict_image_improved.py
Prediction script for improved model with PCA.
"""

import argparse
import json
import joblib
import cv2
import numpy as np
from pathlib import Path
from feature_extraction import extract_features


def load_class_map(path):
    if path is None:
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            if 'class_map' in data:
                class_map = data['class_map']
            else:
                class_map = data
            return {int(k): v for k, v in class_map.items()}
    except Exception as e:
        print(f'Warning: cannot load class_map: {e}')
    return None


def predict_image(image_path, model, scaler, pca, class_map=None, img_size=(128, 128)):
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to RGB and resize
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, img_size)
    
    # Extract features
    features = extract_features(resized)
    features = features.reshape(1, -1)
    
    # Apply preprocessing pipeline
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    
    # Predict
    pred_idx = int(model.predict(features_pca)[0])
    
    # Get probability
    confidence = None
    try:
        probs = model.predict_proba(features_pca)
        confidence = float(np.max(probs))
    except:
        confidence = None
    
    # Get class name
    class_name = class_map.get(pred_idx, str(pred_idx)) if class_map else str(pred_idx)
    
    return pred_idx, class_name, confidence


def main():
    parser = argparse.ArgumentParser(description='Predict with improved PCA model')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='models/svm_model.pkl', help='Path to model')
    parser.add_argument('--scaler', type=str, default='models/scaler.pkl', help='Path to scaler')
    parser.add_argument('--pca', type=str, default='models/pca.pkl', help='Path to PCA')
    parser.add_argument('--class-map', type=str, default='models/training_report_improved.json',
                        help='Path to class map JSON')
    parser.add_argument('--show', action='store_true', help='Display image with prediction')
    
    args = parser.parse_args()
    
    # Load model and preprocessing
    print(f"Loading model from: {args.model}")
    model = joblib.load(args.model)
    
    print(f"Loading scaler from: {args.scaler}")
    scaler = joblib.load(args.scaler)
    
    print(f"Loading PCA from: {args.pca}")
    pca = joblib.load(args.pca)
    
    # Load class map
    class_map = load_class_map(args.class_map)
    if class_map:
        print(f"Class map: {class_map}")
    
    # Predict
    print(f"\nProcessing: {args.image}")
    pred_idx, class_name, confidence = predict_image(
        args.image, model, scaler, pca, class_map
    )
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class: {class_name}")
    print(f"Class Index: {pred_idx}")
    if confidence is not None:
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("="*60)
    
    # Show image
    if args.show:
        img = cv2.imread(args.image)
        if img is not None:
            text = f"Prediction: {class_name}"
            if confidence is not None:
                text += f" ({confidence*100:.1f}%)"
            
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)
            
            cv2.imshow('Prediction (press any key to close)', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
