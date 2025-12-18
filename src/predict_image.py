"""
predict_image.py
Test a trained model on a single image file.
Useful for testing screenshots or individual images.
"""

import argparse
import json
import joblib
import cv2
import numpy as np
from pathlib import Path
from feature_extraction import extract_features
from utils import predict_with_rejection


def load_class_map(path):
    """Load class map from JSON file."""
    if path is None:
        return None
    try:
        with open(path, 'r') as f:
            j = json.load(f)
            if isinstance(j, dict):
                return {int(k): v for k, v in j.items()}
    except Exception as e:
        print(f'Warning: cannot load class_map from {path}: {e}')
    return None


def predict_image(image_path, model, scaler=None, class_map=None, img_size=(128, 128)):
    """
    Predict the class of a single image.
    
    Args:
        image_path: Path to the image file
        model: Trained model
        scaler: Feature scaler (optional)
        class_map: Dictionary mapping class indices to names
        img_size: Size to resize image
    
    Returns:
        Predicted class index, class name, and confidence/probability
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    resized = cv2.resize(rgb, img_size)
    
    # Extract features
    features = extract_features(resized)
    features = features.reshape(1, -1)
    
    # Scale features if scaler provided
    if scaler is not None:
        features = scaler.transform(features)
    
    result = predict_with_rejection(model, features)

    if result.label_idx is None:
        return None, 'Unknown', result.confidence

    pred_idx = int(result.label_idx)
    class_name = class_map.get(pred_idx, str(pred_idx)) if class_map else str(pred_idx)
    return pred_idx, class_name, result.confidence


def main():
    parser = argparse.ArgumentParser(description='Predict class for a single image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pkl)')
    parser.add_argument('--scaler', type=str, default=None,
                        help='Path to scaler (.pkl, optional)')
    parser.add_argument('--class-map', type=str, default=None,
                        help='Path to class_map.json')
    parser.add_argument('--img-size', type=int, nargs=2, default=[128, 128],
                        help='Image size for feature extraction (width height)')
    parser.add_argument('--show', action='store_true',
                        help='Display the image with prediction')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = joblib.load(args.model)
    print(f"Model loaded: {type(model).__name__}")
    
    # Load scaler
    scaler = None
    if args.scaler:
        print(f"Loading scaler from: {args.scaler}")
        scaler = joblib.load(args.scaler)
    
    # Load class map
    class_map = load_class_map(args.class_map)
    if class_map:
        print(f"Class map loaded: {class_map}")
    
    # Predict
    print(f"\nProcessing image: {args.image}")
    pred_idx, class_name, confidence = predict_image(
        args.image, model, scaler, class_map, tuple(args.img_size)
    )
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class: {class_name}")
    print(f"Class Index: {pred_idx if pred_idx is not None else 'N/A'}")
    if confidence is not None:
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("="*60)
    
    # Show image if requested
    if args.show:
        img = cv2.imread(args.image)
        if img is not None:
            # Add prediction text to image
            text = f"Prediction: {class_name}"
            if confidence is not None:
                text += f" ({confidence*100:.1f}%)"
            
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 255, 0), 2)
            
            cv2.imshow('Prediction Result (press any key to close)', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
