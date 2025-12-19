"""
Optimized Training for High Accuracy
Uses augmented dataset + ensemble methods + advanced feature engineering
"""
import numpy as np
import joblib
import json
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import argparse


def load_features(features_file):
    """Load features from npz file"""
    print(f"Loading features from: {features_file}")
    data = np.load(features_file, allow_pickle=True)
    
    feats = data['feats']
    labels = data['labels']
    class_map = json.loads(str(data['class_map']))
    
    print(f"Loaded {feats.shape[0]} samples with {feats.shape[1]} features")
    print(f"Classes: {class_map}")
    
    return feats, labels, class_map


def train_optimized_svm(X_train, y_train, X_val, y_val):
    """Train optimized SVM with best parameters"""
    print("\n" + "="*60)
    print("Training Optimized SVM")
    print("="*60)
    
    # Use best parameters found from previous experiments
    print("Training with optimized parameters (C=50, gamma=0.001)...")
    svm = SVC(C=50, gamma=0.001, kernel='rbf', probability=True, 
              cache_size=1000, class_weight='balanced')
    
    svm.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_acc = svm.score(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return svm


def train_optimized_knn(X_train, y_train, X_val, y_val):
    """Train optimized KNN"""
    print("\n" + "="*60)
    print("Training Optimized KNN")
    print("="*60)
    
    print("Training with optimized parameters (k=9, weights=distance)...")
    knn = KNeighborsClassifier(n_neighbors=9, weights='distance', metric='euclidean')
    
    knn.fit(X_train, y_train)
    
    val_acc = knn.score(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return knn


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest"""
    print("\n" + "="*60)
    print("Training Random Forest")
    print("="*60)
    
    print("Training with optimized parameters...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=2,
                                random_state=42, class_weight='balanced', n_jobs=-1)
    
    rf.fit(X_train, y_train)
    
    val_acc = rf.score(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return rf


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting"""
    print("\n" + "="*60)
    print("Training Gradient Boosting")
    print("="*60)
    
    print("Training with optimized parameters...")
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, 
                                   max_depth=5, random_state=42)
    
    gb.fit(X_train, y_train)
    
    val_acc = gb.score(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return gb


def create_ensemble(svm, knn, rf, gb):
    """Create voting ensemble"""
    print("\n" + "="*60)
    print("Creating Ensemble Model")
    print("="*60)
    
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm),
            ('knn', knn),
            ('rf', rf),
            ('gb', gb)
        ],
        voting='soft',
        weights=[2, 1, 2, 2]  
    )
    
    return ensemble


def evaluate_model(model, X, y, class_map, dataset_name="Test"):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print(f"Evaluating on {dataset_name} Set")
    print("="*60)
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred, target_names=[class_map[i] for i in sorted(class_map.keys())]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train optimized models for high accuracy')
    parser.add_argument('--features', type=str, default='features_optimized.npz',
                        help='Features file')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models')
    
    args = parser.parse_args()
    
    # Load features
    feats, labels, class_map = load_features(args.features)
    
    # Split data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        feats, labels, test_size=0.15, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 of 85% = 15% of total
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Train individual models
    start_time = time.time()
    
    svm = train_optimized_svm(X_train, y_train, X_val, y_val)
    knn = train_optimized_knn(X_train, y_train, X_val, y_val)
    rf = train_random_forest(X_train, y_train, X_val, y_val)
    gb = train_gradient_boosting(X_train, y_train, X_val, y_val)
    
    # Create and train ensemble
    ensemble = create_ensemble(svm, knn, rf, gb)
    
    print("\nTraining ensemble model...")
    ensemble.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")
    
    # Evaluate all models
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    models = {
        'SVM': svm,
        'KNN': knn,
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'Ensemble': ensemble
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n{'='*20} {name} {'='*20}")
        acc = evaluate_model(model, X_test, y_test, class_map, "Test")
        results[name] = acc
    
    # Print summary
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:20s}: {acc*100:6.2f}%")
    
    # Save best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_acc = results[best_model_name]
    
    print(f"\nBest model: {best_model_name} ({best_acc*100:.2f}%)")
    
    # Save models
    print(f"\nSaving models to {args.output_dir}/")
    joblib.dump(svm, f'{args.output_dir}/svm_optimized.pkl')
    joblib.dump(knn, f'{args.output_dir}/knn_optimized.pkl')
    joblib.dump(rf, f'{args.output_dir}/rf_model.pkl')
    joblib.dump(gb, f'{args.output_dir}/gb_model.pkl')
    joblib.dump(ensemble, f'{args.output_dir}/ensemble_model.pkl')
    joblib.dump(scaler, f'{args.output_dir}/scaler_optimized.pkl')
    
    with open(f'{args.output_dir}/class_map.json', 'w') as f:
        json.dump(class_map, f, indent=4)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {best_acc*100:.2f}%")
    print(f"\nSaved models:")
    print(f"  - svm_optimized.pkl")
    print(f"  - knn_optimized.pkl") 
    print(f"  - rf_model.pkl")
    print(f"  - gb_model.pkl")
    print(f"  - ensemble_model.pkl (RECOMMENDED)")
    print(f"  - scaler_optimized.pkl")
    print(f"  - class_map.json")


if __name__ == '__main__':
    main()
