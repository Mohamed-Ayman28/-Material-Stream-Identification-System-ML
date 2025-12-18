"""
compare_models.py - Test image with KNN, SVM, and Ensemble models

Usage: python src\compare_models.py --image <image_path>
"""

import argparse
import cv2
import numpy as np
import joblib
from pathlib import Path

from utils import predict_with_rejection

# Import feature extraction
try:
    from feature_extraction_enhanced import extract_features
    USING_ENHANCED = True
except ImportError:
    from feature_extraction import extract_features
    USING_ENHANCED = False


def predict_with_model(image_path, model_path, scaler_path, class_map, model_name):
    """
    Predict material using a specific model
    
    Args:
        image_path: Path to image
        model_path: Path to model file
        scaler_path: Path to scaler
        class_map: Class mapping dictionary
        model_name: Name of model for display
    
    Returns:
        Prediction results dictionary
    """
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load and preprocess image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f'Could not load image: {image_path}')
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (128, 128))
    
    # Extract features
    features = extract_features(resized)
    features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    result = predict_with_rejection(model, features_scaled)

    confidence = result.confidence
    is_unknown = result.label_idx is None
    if is_unknown:
        class_name = 'Unknown'
    else:
        class_name = class_map.get(int(result.label_idx), str(result.label_idx))

    all_probas = None
    if result.probabilities is not None:
        all_probas = {class_map.get(int(i), str(i)): float(p) for i, p in result.probabilities.items()}
    
    return {
        'model_name': model_name,
        'prediction': class_name,
        'confidence': confidence,
        'probabilities': all_probas,
        'is_unknown': is_unknown
    }


def compare_models(image_path):
    """
    Compare predictions from KNN, SVM, and Ensemble models
    
    Args:
        image_path: Path to image to test
    """
    # Load class map
    import json
    with open('models/class_map_enhanced.json', 'r') as f:
        class_map_data = json.load(f)
    class_map = {int(k): v for k, v in class_map_data.items()}
    
    print("\n" + "="*80)
    print(f"COMPARING MODELS ON: {image_path}")
    print("="*80)
    
    # Define models to test
    models = [
        {
            'name': 'k-NN',
            'path': 'models/knn_enhanced.pkl',
            'scaler': 'models/scaler_enhanced.pkl'
        },
        {
            'name': 'SVM',
            'path': 'models/svm_enhanced.pkl',
            'scaler': 'models/scaler_enhanced.pkl'
        },
        {
            'name': 'Ensemble',
            'path': 'models/ensemble_enhanced.pkl',
            'scaler': 'models/scaler_enhanced.pkl'
        }
    ]
    
    results = []
    
    # Test each model
    for model_info in models:
        try:
            result = predict_with_model(
                image_path,
                model_info['path'],
                model_info['scaler'],
                class_map,
                model_info['name']
            )
            results.append(result)
        except Exception as e:
            print(f"\nError testing {model_info['name']}: {e}")
            continue
    
    # Display results
    print("\n" + "-"*80)
    print("PREDICTION RESULTS")
    print("-"*80)
    
    for result in results:
        print(f"\n{result['model_name']}:")
        print(f"  Prediction: {result['prediction']}")
        if result['confidence'] is not None:
            print(f"  Confidence: {result['confidence']:.2%}")
        if result['is_unknown']:
            print(f"  Status: Unknown/Uncertain")
        
        if result['probabilities']:
            print(f"  Top 3 probabilities:")
            sorted_probs = sorted(result['probabilities'].items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
            for cls, prob in sorted_probs:
                print(f"    {cls:12s}: {prob:.2%}")
    
    # Summary comparison
    print("\n" + "-"*80)
    print("CONSENSUS ANALYSIS")
    print("-"*80)
    
    predictions = [r['prediction'] for r in results]
    confidences = [r['confidence'] for r in results if r['confidence'] is not None]
    
    # Check if all models agree
    if len(set(predictions)) == 1:
        print(f"  All models agree: {predictions[0]}")
        print(f"  Average confidence: {np.mean(confidences):.2%}")
    else:
        print(f"  Models disagree:")
        for result in results:
            conf_str = f" ({result['confidence']:.2%})" if result['confidence'] else ""
            print(f"    {result['model_name']:10s} -> {result['prediction']}{conf_str}")
    
    # Recommendation
    print("\n  Recommendation:")
    ensemble_result = [r for r in results if r['model_name'] == 'Ensemble'][0]
    if ensemble_result['is_unknown']:
        print("    Item is uncertain - classified as Unknown")
    else:
        print(f"    Use Ensemble prediction: {ensemble_result['prediction']}")
        print(f"    Confidence: {ensemble_result['confidence']:.2%}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare KNN, SVM, and Ensemble predictions')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image to test')
    
    args = parser.parse_args()
    
    # Check image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    # Run comparison
    compare_models(args.image)


if __name__ == '__main__':
    main()
