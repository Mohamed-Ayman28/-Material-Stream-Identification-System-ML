"""
train_knn_svm.py - Optimized training for KNN and SVM models
Focuses on traditional ML without deep learning for material classification
"""

import argparse
import time
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


def load_features(feature_file):
    """Load features from npz file"""
    print(f'Loading features from {feature_file}...')
    data = np.load(feature_file)
    X = data['features']
    y = data['labels']
    print(f'Loaded {X.shape[0]} samples with {X.shape[1]} features each')
    print(f'Class distribution: {np.bincount(y)}')
    return X, y


def train_optimized_svm(X_train, y_train, X_val, y_val):
    """Train SVM with optimized hyperparameters"""
    print('\n=== Training SVM ===')
    
    # Grid search for best hyperparameters
    param_grid = {
        'C': [10, 50, 100],
        'gamma': ['scale', 0.001, 0.01],
        'kernel': ['rbf']
    }
    
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    best_svm = grid_search.best_estimator_
    print(f'Best SVM parameters: {grid_search.best_params_}')
    print(f'Training time: {training_time:.2f}s')
    
    # Evaluate
    train_score = best_svm.score(X_train, y_train)
    val_score = best_svm.score(X_val, y_val)
    
    print(f'Training accuracy: {train_score*100:.2f}%')
    print(f'Validation accuracy: {val_score*100:.2f}%')
    
    return best_svm, val_score


def train_optimized_knn(X_train, y_train, X_val, y_val):
    """Train KNN with optimized hyperparameters"""
    print('\n=== Training KNN ===')
    
    # Grid search for best k and weights
    param_grid = {
        'n_neighbors': [5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    best_knn = grid_search.best_estimator_
    print(f'Best KNN parameters: {grid_search.best_params_}')
    print(f'Training time: {training_time:.2f}s')
    
    # Evaluate
    train_score = best_knn.score(X_train, y_train)
    val_score = best_knn.score(X_val, y_val)
    
    print(f'Training accuracy: {train_score*100:.2f}%')
    print(f'Validation accuracy: {val_score*100:.2f}%')
    
    return best_knn, val_score


def create_ensemble(svm_model, knn_model, X_train, y_train, X_val, y_val):
    """Create ensemble model using soft voting"""
    print('\n=== Creating Ensemble Model ===')
    
    # Weight based on validation performance
    svm_weight = svm_model.score(X_val, y_val)
    knn_weight = knn_model.score(X_val, y_val)
    
    # Normalize weights
    total_weight = svm_weight + knn_weight
    svm_weight = svm_weight / total_weight
    knn_weight = knn_weight / total_weight
    
    print(f'SVM weight: {svm_weight:.3f}, KNN weight: {knn_weight:.3f}')
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('knn', knn_model)
        ],
        voting='soft',
        weights=[svm_weight, knn_weight]
    )
    
    start_time = time.time()
    ensemble.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f'Ensemble training time: {training_time:.2f}s')
    
    # Evaluate
    train_score = ensemble.score(X_train, y_train)
    val_score = ensemble.score(X_val, y_val)
    
    print(f'Training accuracy: {train_score*100:.2f}%')
    print(f'Validation accuracy: {val_score*100:.2f}%')
    
    return ensemble, val_score


def evaluate_model(model, X_test, y_test, class_map, model_name='Model'):
    """Detailed evaluation of model"""
    print(f'\n=== Evaluating {model_name} on Test Set ===')
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    print('\nClassification Report:')
    
    target_names = [class_map[i] for i in sorted(class_map.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print('\nConfusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train optimized KNN and SVM models')
    parser.add_argument('--features', required=True, help='Path to features npz file')
    parser.add_argument('--output-dir', default='models', help='Directory to save models')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load features
    X, y = load_features(args.features)
    
    # Create class map
    class_map = {
        0: 'glass',
        1: 'paper',
        2: 'cardboard',
        3: 'plastic',
        4: 'metal',
        5: 'trash'
    }
    
    # Split data
    print(f'\nSplitting data: test={args.test_size}, val={args.val_size}')
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )
    
    val_ratio = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=args.random_seed, stratify=y_temp
    )
    
    print(f'Train set: {X_train.shape[0]} samples')
    print(f'Validation set: {X_val.shape[0]} samples')
    print(f'Test set: {X_test.shape[0]} samples')
    
    # Scale features
    print('\nScaling features...')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    total_start = time.time()
    
    svm_model, svm_val_score = train_optimized_svm(X_train_scaled, y_train, X_val_scaled, y_val)
    knn_model, knn_val_score = train_optimized_knn(X_train_scaled, y_train, X_val_scaled, y_val)
    ensemble_model, ensemble_val_score = create_ensemble(
        svm_model, knn_model, X_train_scaled, y_train, X_val_scaled, y_val
    )
    
    total_time = time.time() - total_start
    print(f'\nTotal training time: {total_time:.2f}s')
    
    # Evaluate on test set
    svm_test_acc = evaluate_model(svm_model, X_test_scaled, y_test, class_map, 'SVM')
    knn_test_acc = evaluate_model(knn_model, X_test_scaled, y_test, class_map, 'KNN')
    ensemble_test_acc = evaluate_model(ensemble_model, X_test_scaled, y_test, class_map, 'Ensemble')
    
    # Save models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f'\nSaving models to {output_dir}...')
    joblib.dump(svm_model, output_dir / 'svm_model.pkl')
    joblib.dump(knn_model, output_dir / 'knn_model.pkl')
    joblib.dump(ensemble_model, output_dir / 'ensemble_knn_svm.pkl')
    joblib.dump(scaler, output_dir / 'scaler.pkl')
    
    # Save class map
    with open(output_dir / 'class_map.json', 'w') as f:
        json.dump(class_map, f, indent=2)
    
    # Save training report
    report = {
        'training_time': total_time,
        'class_map': class_map,
        'models': {
            'svm': {
                'validation_accuracy': float(svm_val_score),
                'test_accuracy': float(svm_test_acc),
                'params': svm_model.get_params()
            },
            'knn': {
                'validation_accuracy': float(knn_val_score),
                'test_accuracy': float(knn_test_acc),
                'params': knn_model.get_params()
            },
            'ensemble': {
                'validation_accuracy': float(ensemble_val_score),
                'test_accuracy': float(ensemble_test_acc)
            }
        },
        'dataset': {
            'total_samples': int(X.shape[0]),
            'features': int(X.shape[1]),
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'test_samples': int(X_test.shape[0])
        }
    }
    
    with open(output_dir / 'training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print('\n=== Training Complete ===')
    print(f'SVM test accuracy: {svm_test_acc*100:.2f}%')
    print(f'KNN test accuracy: {knn_test_acc*100:.2f}%')
    print(f'Ensemble test accuracy: {ensemble_test_acc*100:.2f}%')
    print(f'\nModels saved to: {output_dir}')


if __name__ == '__main__':
    main()
