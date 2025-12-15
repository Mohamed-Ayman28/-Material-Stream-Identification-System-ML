"""
main.py
Main entry point for the Material Stream Identification System.
This script orchestrates the entire pipeline:
1. Data augmentation
2. Feature extraction
3. Model training
4. Model evaluation
5. Deployment preparation
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from augmentation import combinedAugmentation
from feature_extraction import extract_features
from train import load_features, scale_features, train_svm_grid, train_knn_grid 
from evaluate import evaluate_model
import joblib
import json


def augment_dataset(input_dir, output_dir, num_augmentations=5):
    """
    Augment the entire dataset by applying random transformations.
    
    Args:
        input_dir: Path to original dataset directory
        output_dir: Path to save augmented images
        num_augmentations: Number of augmented versions per image
    """
    print(f"Starting data augmentation...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return
    
    # Get all class directories
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    total_images = 0
    total_augmented = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Create output directory for this class
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images in this class
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        
        for img_file in image_files:
            total_images += 1
            
            # Copy original image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Could not read {img_file}")
                continue
            
            # Save original
            original_name = output_class_dir / img_file.name
            cv2.imwrite(str(original_name), img)
            
            # Generate augmented versions
            for i in range(num_augmentations):
                aug_img = combinedAugmentation(img.copy())
                aug_name = output_class_dir / f"{img_file.stem}_aug{i}{img_file.suffix}"
                cv2.imwrite(str(aug_name), aug_img)
                total_augmented += 1
        
        print(f"  - Processed {len(image_files)} images")
    
    print(f"\nAugmentation complete!")
    print(f"Total original images: {total_images}")
    print(f"Total augmented images: {total_augmented}")
    print(f"Total images in output: {total_images + total_augmented}")


def extract_dataset_features(dataset_dir, output_npz, img_size=(128, 128)):
    """
    Extract features from all images in the dataset.
    
    Args:
        dataset_dir: Path to dataset directory
        output_npz: Path to save extracted features
        img_size: Target image size for resizing
    """
    print(f"\nExtracting features from dataset...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output file: {output_npz}")
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist!")
        return
    
    # Define correct class mapping according to specification
    # ID 0: Glass, ID 1: Paper, ID 2: Cardboard, ID 3: Plastic, ID 4: Metal, ID 5: Trash
    correct_class_mapping = {
        'glass': 0,
        'paper': 1,
        'cardboard': 2,
        'plastic': 3,
        'metal': 4,
        'trash': 5
    }
    
    # Create reverse mapping for saving
    class_map = {v: k for k, v in correct_class_mapping.items()}
    
    all_features = []
    all_labels = []
    
    # Process each class directory according to the correct mapping
    for class_name, class_id in sorted(correct_class_mapping.items(), key=lambda x: x[1]):
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory '{class_name}' not found, skipping...")
            continue
            
        print(f"\nProcessing class {class_id}: {class_name}")
        
        # Get all images
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Resize image
            img_resized = cv2.resize(img, img_size)
            
            # Extract features
            features = extract_features(img_resized)
            
            all_features.append(features)
            all_labels.append(class_id)
        
        print(f"  - Extracted features from {len(image_files)} images")
    
    # Convert to numpy arrays
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    print(f"\nFeature extraction complete!")
    print(f"Feature shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print(f"Class mapping: {class_map}")
    
    # Save to npz file
    np.savez(
        output_npz,
        feats=all_features,
        labels=all_labels,
        class_map=json.dumps(class_map)
    )
    
    print(f"Features saved to: {output_npz}")
    
    return all_features, all_labels, class_map


def main():
    parser = argparse.ArgumentParser(description='Material Stream Identification System - Main Pipeline')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['augment', 'extract', 'train', 'full'],
                        help='Mode: augment (data augmentation), extract (feature extraction), train (model training), or full (complete pipeline)')
    parser.add_argument('--input-dir', type=str, default='dataset',
                        help='Input dataset directory')
    parser.add_argument('--augmented-dir', type=str, default='dataset_augmented',
                        help='Directory to save augmented dataset')
    parser.add_argument('--num-aug', type=int, default=5,
                        help='Number of augmented versions per image')
    parser.add_argument('--features-file', type=str, default='features.npz',
                        help='Path to save/load extracted features')
    parser.add_argument('--model-type', type=str, default='svm', choices=['svm', 'knn', 'both'],
                        help='Model type to train')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--img-size', type=int, nargs=2, default=[128, 128],
                        help='Image size for feature extraction (width height)')
    parser.add_argument('--fast', action='store_true',
                        help='Use fast training mode (smaller grid search, faster but potentially less accurate)')
    parser.add_argument('--cv-folds', type=int, default=3,
                        help='Number of cross-validation folds (reduce to 2 for faster training)')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'augment' or args.mode == 'full':
        print("="*60)
        print("STEP 1: DATA AUGMENTATION")
        print("="*60)
        augment_dataset(args.input_dir, args.augmented_dir, args.num_aug)
    
    if args.mode == 'extract' or args.mode == 'full':
        print("\n" + "="*60)
        print("STEP 2: FEATURE EXTRACTION")
        print("="*60)
        
        # Use augmented dataset if in full mode, otherwise use specified input
        dataset_to_use = args.augmented_dir if args.mode == 'full' else args.input_dir
        extract_dataset_features(dataset_to_use, args.features_file, tuple(args.img_size))
    
    if args.mode == 'train' or args.mode == 'full':
        print("\n" + "="*60)
        print("STEP 3: MODEL TRAINING")
        print("="*60)
        
        # Load features
        print(f"\nLoading features from {args.features_file}...")
        feats, labels, class_map = load_features(args.features_file)
        print(f"Loaded {len(feats)} samples with {feats.shape[1]} features")
        print(f"Classes: {class_map}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            feats, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\nDataset split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Scale features
        scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test) #ignore
        
        # Save scaler
        scaler_path = Path(args.model_dir) / 'scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"\nScaler saved to: {scaler_path}")
        
        # Train models
        if args.model_type in ['svm', 'both']:
            print("\n" + "-"*60)
            print("Training SVM model...")
            if args.fast:
                print("[FAST MODE: Using reduced parameter grid]")
            print("-"*60)
            svm_model, svm_params, _ = train_svm_grid(X_train_scaled, y_train, cv=args.cv_folds, n_jobs=-1, verbose=1, fast_mode=args.fast)
            print(f"Best SVM parameters: {svm_params}")
            
            # Evaluate
            print("\nSVM Model Evaluation:")
            svm_val_results, _, _ = evaluate_model(svm_model, X_val_scaled, y_val, "Validation", class_map)
            svm_test_results, _, _ = evaluate_model(svm_model, X_test_scaled, y_test, "Test", class_map)
            
            # Save model
            svm_path = Path(args.model_dir) / 'svm_model.pkl'
            joblib.dump(svm_model, svm_path)
            print(f"\nSVM model saved to: {svm_path}")
        
        if args.model_type in ['knn', 'both']:
            print("\n" + "-"*60)
            print("Training KNN model...")
            print("-"*60)
            knn_model, knn_params = train_knn_grid(X_train_scaled, y_train, cv=args.cv_folds)
            print(f"Best KNN parameters: {knn_params}")
            
            # Evaluate
            print("\nKNN Model Evaluation:")
            knn_val_results, _, _ = evaluate_model(knn_model, X_val_scaled, y_val, "Validation", class_map)
            knn_test_results, _, _ = evaluate_model(knn_model, X_test_scaled, y_test, "Test", class_map)
            
            # Save model
            knn_path = Path(args.model_dir) / 'knn_model.pkl'
            joblib.dump(knn_model, knn_path)
            print(f"\nKNN model saved to: {knn_path}")
        
        # Save class map
        class_map_path = Path(args.model_dir) / 'class_map.json'
        with open(class_map_path, 'w') as f:
            json.dump(class_map, f, indent=2)
        print(f"\nClass map saved to: {class_map_path}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nAll models and artifacts saved in: {args.model_dir}/")
        print("\nYou can now use deploy.py for real-time inference.")


if __name__ == '__main__':
    main()
