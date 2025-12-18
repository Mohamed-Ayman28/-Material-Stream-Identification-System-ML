"""
retrain_enhanced.py - Retrain models with enhanced features and better parameters
This improves accuracy WITHOUT collecting new data
"""

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
import cv2
import os
from tqdm import tqdm
import json
from feature_extraction_enhanced import extract_features


def load_dataset():
    """Load and process the existing dataset with enhanced features"""
    print("\n" + "="*60)
    print("LOADING DATASET WITH ENHANCED FEATURES")
    print("="*60)
    
    dataset_path = 'dataset'
    classes = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
    class_map = {i: cls for i, cls in enumerate(classes)}
    
    X = []
    y = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️  Warning: {class_path} not found, skipping...")
            continue
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\n📂 Processing {class_name}: {len(images)} images")
        
        for img_file in tqdm(images, desc=f"  Extracting {class_name}"):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Preprocess
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (128, 128))
            
            # Extract enhanced features
            features = extract_features(resized)
            
            X.append(features)
            y.append(class_idx)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"\n✅ Dataset loaded: {len(X)} samples")
    print(f"   Feature dimensions: {X.shape[1]}")
    print(f"   Class distribution:")
    for i, cls in class_map.items():
        count = np.sum(y == i)
        print(f"      {cls}: {count}")
    
    return X, y, class_map


def train_optimized_svm(X_train, y_train):
    """Train SVM with expanded parameter grid for better cardboard detection"""
    print("\n" + "="*60)
    print("TRAINING SVM WITH ENHANCED PARAMETERS")
    print("="*60)
    
    # Broader parameter search
    param_grid = {
        'C': [1, 10, 50, 100, 500],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly'],
        'class_weight': ['balanced', None]
    }
    
    svm = SVC(probability=True, random_state=42)
    
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    
    print("🔍 Performing grid search (this may take several minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best SVM parameters: {grid_search.best_params_}")
    print(f"   Best cross-val score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_optimized_knn(X_train, y_train):
    """Train KNN with expanded parameters"""
    print("\n" + "="*60)
    print("TRAINING KNN WITH ENHANCED PARAMETERS")
    print("="*60)
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2, 3]
    }
    
    knn = KNeighborsClassifier()
    
    grid_search = GridSearchCV(
        knn, param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    
    print("🔍 Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best KNN parameters: {grid_search.best_params_}")
    print(f"   Best cross-val score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def create_enhanced_ensemble(svm_model, knn_model, X_train, y_train):
    """Create ensemble with better weighting"""
    print("\n" + "="*60)
    print("CREATING ENHANCED ENSEMBLE")
    print("="*60)
    
    # Test both models
    svm_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
    knn_scores = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='accuracy')
    
    svm_weight = svm_scores.mean()
    knn_weight = knn_scores.mean()
    
    print(f"  SVM average accuracy: {svm_weight:.4f}")
    print(f"  KNN average accuracy: {knn_weight:.4f}")
    
    # Create weighted ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('knn', knn_model)
        ],
        voting='soft',
        weights=[svm_weight, knn_weight]
    )
    
    print("  Training ensemble...")
    ensemble.fit(X_train, y_train)
    
    ensemble_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  Ensemble average accuracy: {ensemble_scores.mean():.4f}")
    
    return ensemble


def main():
    """Main retraining workflow"""
    print("\n" + "="*60)
    print("ENHANCED MODEL RETRAINING")
    print("Improving accuracy WITHOUT collecting new data")
    print("="*60)
    
    # Load dataset with enhanced features
    X, y, class_map = load_dataset()
    
    # Scale features
    print("\n📊 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train models
    svm_model = train_optimized_svm(X_scaled, y)
    knn_model = train_optimized_knn(X_scaled, y)
    ensemble_model = create_enhanced_ensemble(svm_model, knn_model, X_scaled, y)
    
    # Save models
    print("\n" + "="*60)
    print("SAVING ENHANCED MODELS")
    print("="*60)
    
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(svm_model, 'models/svm_enhanced.pkl')
    print("✅ Saved: models/svm_enhanced.pkl")
    
    joblib.dump(knn_model, 'models/knn_enhanced.pkl')
    print("✅ Saved: models/knn_enhanced.pkl")
    
    joblib.dump(ensemble_model, 'models/ensemble_enhanced.pkl')
    print("✅ Saved: models/ensemble_enhanced.pkl")
    
    joblib.dump(scaler, 'models/scaler_enhanced.pkl')
    print("✅ Saved: models/scaler_enhanced.pkl")
    
    with open('models/class_map_enhanced.json', 'w') as f:
        json.dump(class_map, f, indent=2)
    print("✅ Saved: models/class_map_enhanced.json")
    
    print("\n" + "="*60)
    print("✅ RETRAINING COMPLETE!")
    print("="*60)
    print("\nEnhanced models are ready to use.")
    print("Test with: python src\\predict.py --image <image_path> --model models\\ensemble_enhanced.pkl --scaler models\\scaler_enhanced.pkl")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
