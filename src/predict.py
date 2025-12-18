"""
predict.py - Simple prediction script for testing material classification
Uses trained KNN and SVM models
"""

import argparse
import json
import numpy as np
from pathlib import Path
import cv2
import joblib
# Rejection mechanism (model-specific)
from utils import predict_with_rejection
# Use enhanced features (87-feature version) for *_enhanced models
try:
    from feature_extraction_enhanced import extract_features
    USING_ENHANCED = True
except ImportError:
    from feature_extraction import extract_features
    USING_ENHANCED = False


def load_class_map(path):
    """Load class mapping from JSON file"""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'class_map' in data:
                return {int(k): v for k, v in data['class_map'].items()}
            return {int(k): v for k, v in data.items()}
    except Exception as e:
        print(f'Warning: Could not load class map: {e}')
        return None


def predict_image(image_path, model, scaler=None, class_map=None, img_size=(128, 128)):
    """
    Predict material class for a single image
    
    Args:
        image_path: Path to image file
        model: Trained classification model
        scaler: Feature scaler
        class_map: Dictionary mapping class indices to names
        img_size: Size to resize image for feature extraction
        
    Returns:
        Predicted class, confidence score
    """
    # Load and preprocess image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f'Could not load image: {image_path}')
    
    # Convert to RGB and resize
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, img_size)
    
    # Extract features
    features = extract_features(resized)
    features = features.reshape(1, -1)
    
    # Scale features if scaler provided
    if scaler is not None:
        features = scaler.transform(features)
    
    # Predict with model-specific rejection
    result = predict_with_rejection(model, features)

    # Convert probabilities (if available) to class-name map
    all_probas = None
    if result.probabilities is not None:
        all_probas = {
            (class_map.get(int(i), str(i)) if class_map else str(i)): float(p)
            for i, p in result.probabilities.items()
        }
        if result.label_idx is None and result.confidence is not None:
            all_probas['Unknown'] = float(result.confidence)

    confidence = result.confidence

    if result.label_idx is None:
        return 'Unknown', confidence, all_probas

    class_name = class_map.get(int(result.label_idx), str(result.label_idx)) if class_map else str(result.label_idx)
    return class_name, confidence, all_probas


def main():
    parser = argparse.ArgumentParser(description='Predict material class from image')
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--model', required=True, help='Path to trained model (pkl)')
    parser.add_argument('--scaler', help='Path to scaler file (pkl)')
    parser.add_argument('--class-map', help='Path to class map JSON file')
    parser.add_argument('--width', type=int, default=128, help='Image width for processing')
    parser.add_argument('--height', type=int, default=128, help='Image height for processing')
    
    args = parser.parse_args()
    
    # Load model
    print(f'Loading model from {args.model}...')
    model = joblib.load(args.model)
    
    # Load scaler
    scaler = None
    if args.scaler:
        print(f'Loading scaler from {args.scaler}...')
        scaler = joblib.load(args.scaler)
    
    # Load class map
    class_map = None
    if args.class_map:
        print(f'Loading class map from {args.class_map}...')
        class_map = load_class_map(args.class_map)
    
    # Predict
    print(f'\nPredicting material for: {args.image}')
    class_name, confidence, all_probas = predict_image(
        args.image, model, scaler, class_map, 
        img_size=(args.width, args.height)
    )
    
    # Display results
    print(f'\n{"="*50}')
    print(f'Predicted Material: {class_name}')
    if confidence is not None:
        print(f'Confidence: {confidence:.2%}')
    if class_name == 'Unknown':
        print('⚠️  Item not recognized or too uncertain to classify')
    print(f'{"="*50}')
    
    if all_probas:
        print('\nAll class probabilities:')
        for class_name, prob in sorted(all_probas.items(), key=lambda x: x[1], reverse=True):
            print(f'  {class_name:12s}: {prob:.2%}')


if __name__ == '__main__':
    main()
