"""
CNN Image Prediction Script
Predict material class for a single image using trained CNN model
"""
import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import argparse


def load_model(model_path):
    """Load trained CNN model"""
    print(f"Loading model from: {model_path}")
    
    if model_path.endswith('.h5'):
        model = keras.models.load_model(model_path)
    else:
        model = keras.models.load_model(model_path)
    
    print(f"Model loaded successfully")
    return model


def load_class_map(class_map_path):
    """Load class mapping"""
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
    
    # Convert keys to int
    class_map = {int(k): v for k, v in class_map.items()}
    print(f"Class map loaded: {class_map}")
    return class_map


def preprocess_image(image_path, img_size=(224, 224)):
    """Load and preprocess image for CNN"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, img_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Preprocess for MobileNetV2
    img = keras.applications.mobilenet_v2.preprocess_input(img * 255.0)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def predict_image(image_path, model, class_map, img_size=(224, 224)):
    """Predict class for a single image"""
    # Preprocess
    img = preprocess_image(image_path, img_size)
    
    # Predict
    predictions = model.predict(img, verbose=0)
    
    # Get top prediction
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx]
    class_name = class_map[pred_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3 = [(class_map[i], predictions[0][i]) for i in top_3_idx]
    
    return pred_idx, class_name, confidence, top_3, predictions[0]


def visualize_prediction(image_path, class_name, confidence, top_3):
    """Display prediction results with image"""
    import matplotlib.pyplot as plt
    
    # Read and display image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show image
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'Predicted: {class_name.upper()} ({confidence*100:.1f}%)', 
                  fontsize=14, fontweight='bold')
    
    # Show top 3 predictions
    classes = [c for c, _ in top_3]
    confidences = [conf * 100 for _, conf in top_3]
    colors = ['green' if i == 0 else 'orange' if i == 1 else 'red' for i in range(3)]
    
    bars = ax2.barh(classes, confidences, color=colors, alpha=0.7)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Top 3 Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Add value labels
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        ax2.text(conf + 2, i, f'{conf:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Predict material class using CNN')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--model', type=str, default='models/cnn_model.h5',
                        help='Path to trained model')
    parser.add_argument('--class-map', type=str, default='models/cnn_class_map.json',
                        help='Path to class mapping JSON')
    parser.add_argument('--img-size', type=int, nargs=2, default=[224, 224],
                        help='Image size (width height)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization of prediction')
    
    args = parser.parse_args()
    
    # Load model and class map
    model = load_model(args.model)
    class_map = load_class_map(args.class_map)
    
    # Predict
    print(f"\nProcessing image: {args.image}")
    pred_idx, class_name, confidence, top_3, all_probs = predict_image(
        args.image, model, class_map, tuple(args.img_size)
    )
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class: {class_name}")
    print(f"Class Index: {pred_idx}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("\nTop 3 Predictions:")
    for i, (cls, conf) in enumerate(top_3, 1):
        print(f"  {i}. {cls}: {conf*100:.2f}%")
    print("="*60)
    
    # Visualize if requested
    if args.visualize:
        visualize_prediction(args.image, class_name, confidence, top_3)


if __name__ == '__main__':
    main()
