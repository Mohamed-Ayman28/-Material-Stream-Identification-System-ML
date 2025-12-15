"""
train_improved.py
Improved training with PCA dimensionality reduction for better performance.
"""

import argparse
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42


def load_features(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    feats = d['feats']
    labels = d['labels']
    class_map = {}
    if 'class_map' in d:
        try:
            class_map = json.loads(d['class_map'].tolist())
        except Exception:
            class_map = d['class_map'].tolist()
    # Convert string keys to int
    class_map = {int(k): v for k, v in class_map.items()}
    return feats, labels, class_map


def main():
    parser = argparse.ArgumentParser(description='Train improved model with PCA')
    parser.add_argument('--features', type=str, default='features.npz',
                        help='Path to features file')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--pca-components', type=int, default=100,
                        help='Number of PCA components (reduce from 8118 to this)')
    parser.add_argument('--model-type', type=str, default='svm',
                        choices=['svm', 'rf', 'both'],
                        help='Model type to train')
    
    args = parser.parse_args()
    
    print("="*60)
    print("IMPROVED TRAINING WITH DIMENSIONALITY REDUCTION")
    print("="*60)
    
    # Load features
    print(f"\nLoading features from {args.features}...")
    X, y, class_map = load_features(args.features)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Classes: {class_map}")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Step 1: Scale features
    print("\n" + "-"*60)
    print("Step 1: Feature Scaling")
    print("-"*60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: Apply PCA for dimensionality reduction
    print("\n" + "-"*60)
    print(f"Step 2: PCA Dimensionality Reduction (8118 -> {args.pca_components})")
    print("-"*60)
    pca = PCA(n_components=args.pca_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA complete: {args.pca_components} components explain {explained_var*100:.2f}% of variance")
    print(f"New feature shape: {X_train_pca.shape}")
    
    # Create model directory
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preprocessing objects
    scaler_path = model_dir / 'scaler.pkl'
    pca_path = model_dir / 'pca.pkl'
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    print(f"\nScaler saved to: {scaler_path}")
    print(f"PCA saved to: {pca_path}")
    
    # Train models
    results = {}
    
    if args.model_type in ['svm', 'both']:
        print("\n" + "="*60)
        print("Training SVM with RBF kernel")
        print("="*60)
        
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 0.001, 0.01]
        }
        
        svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, cache_size=1000)
        grid = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=2)
        
        print("Training (this may take a few minutes)...")
        grid.fit(X_train_pca, y_train)
        
        svm_model = grid.best_estimator_
        print(f"\nBest parameters: {grid.best_params_}")
        
        # Evaluate
        val_pred = svm_model.predict(X_val_pca)
        test_pred = svm_model.predict(X_test_pca)
        
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\nValidation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, test_pred, 
                                    target_names=[class_map[i] for i in sorted(class_map.keys())]))
        
        # Save model
        svm_path = model_dir / 'svm_model.pkl'
        joblib.dump(svm_model, svm_path)
        print(f"\nSVM model saved to: {svm_path}")
        
        results['svm'] = {
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'best_params': grid.best_params_
        }
    
    if args.model_type in ['rf', 'both']:
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)
        
        print("Training (this may take a few minutes)...")
        grid.fit(X_train_pca, y_train)
        
        rf_model = grid.best_estimator_
        print(f"\nBest parameters: {grid.best_params_}")
        
        # Evaluate
        val_pred = rf_model.predict(X_val_pca)
        test_pred = rf_model.predict(X_test_pca)
        
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\nValidation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, test_pred,
                                    target_names=[class_map[i] for i in sorted(class_map.keys())]))
        
        # Save model
        rf_path = model_dir / 'rf_model.pkl'
        joblib.dump(rf_model, rf_path)
        print(f"\nRandom Forest model saved to: {rf_path}")
        
        results['rf'] = {
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'best_params': grid.best_params_
        }
    
    # Save training report
    report = {
        'pca_components': args.pca_components,
        'explained_variance': float(explained_var),
        'class_map': class_map,
        'results': results
    }
    
    report_path = model_dir / 'training_report_improved.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nAll models saved to: {args.model_dir}/")
    print(f"Report saved to: {report_path}")
    print("\nIMPORTANT: Use predict_image_improved.py for predictions with the new model")


if __name__ == '__main__':
    main()
