"""
Enhanced Model Training with Advanced Techniques
- Ensemble methods
- Better hyperparameter tuning
- Class balancing
"""
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import argparse


def load_features(features_file):
    """Load features from npz file"""
    print(f"\nLoading features from {features_file}...")
    data = np.load(features_file, allow_pickle=True)
    
    X = data['feats']
    y = data['labels']
    class_map = json.loads(str(data['class_map']))
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Classes: {class_map}")
    
    return X, y, class_map


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest with grid search"""
    print("\n" + "="*60)
    print("Training Random Forest Classifier")
    print("="*60)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    print("Performing grid search (this may take a while)...")
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy',
        verbose=1, n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation
    y_pred = grid_search.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return grid_search.best_estimator_


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting with grid search"""
    print("\n" + "="*60)
    print("Training Gradient Boosting Classifier")
    print("="*60)
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    
    print("Performing grid search (this may take a while)...")
    grid_search = GridSearchCV(
        gb, param_grid, cv=3, scoring='accuracy',
        verbose=1, n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation
    y_pred = grid_search.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return grid_search.best_estimator_


def train_improved_svm(X_train, y_train, X_val, y_val):
    """Train SVM with extensive grid search"""
    print("\n" + "="*60)
    print("Training Improved SVM Classifier")
    print("="*60)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'poly', 'linear'],
        'gamma': ['scale', 'auto', 0.01, 0.1]
    }
    
    svm = SVC(random_state=42, probability=True)
    
    print("Performing grid search...")
    grid_search = GridSearchCV(
        svm, param_grid, cv=3, scoring='accuracy',
        verbose=1, n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation
    y_pred = grid_search.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return grid_search.best_estimator_


def train_improved_knn(X_train, y_train, X_val, y_val):
    """Train KNN with extensive grid search"""
    print("\n" + "="*60)
    print("Training Improved KNN Classifier")
    print("="*60)
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]
    }
    
    knn = KNeighborsClassifier(n_jobs=-1)
    
    print("Performing grid search...")
    grid_search = GridSearchCV(
        knn, param_grid, cv=3, scoring='accuracy',
        verbose=1, n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation
    y_pred = grid_search.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return grid_search.best_estimator_


def create_ensemble(models):
    """Create voting ensemble of best models"""
    print("\n" + "="*60)
    print("Creating Ensemble Model")
    print("="*60)
    
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',
        n_jobs=-1
    )
    
    return ensemble


def evaluate_model(model, X_test, y_test, class_map, model_name):
    """Evaluate model and print detailed metrics"""
    print("\n" + "="*60)
    print(f"Evaluating {model_name} on Test Set")
    print("="*60)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Get class names
    class_names = [class_map[str(i)] for i in sorted([int(k) for k in class_map.keys()])]
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Enhanced model training')
    parser.add_argument('--features-file', type=str, default='features_optimized.npz',
                        help='Features file')
    parser.add_argument('--model-type', type=str, default='all',
                        choices=['rf', 'gb', 'svm', 'knn', 'ensemble', 'all'],
                        help='Model type to train')
    
    args = parser.parse_args()
    
    # Load data
    X, y, class_map = load_features(args.features_file)
    
    # Split data
    print("\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = []
    model_names = []
    
    if args.model_type in ['rf', 'all']:
        rf_model = train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
        models.append(('rf', rf_model))
        model_names.append('Random Forest')
        
        # Save
        joblib.dump(rf_model, 'models/rf_model.pkl')
        print("Random Forest saved to: models/rf_model.pkl")
    
    if args.model_type in ['gb', 'all']:
        gb_model = train_gradient_boosting(X_train_scaled, y_train, X_val_scaled, y_val)
        models.append(('gb', gb_model))
        model_names.append('Gradient Boosting')
        
        # Save
        joblib.dump(gb_model, 'models/gb_model.pkl')
        print("Gradient Boosting saved to: models/gb_model.pkl")
    
    if args.model_type in ['svm', 'all']:
        svm_model = train_improved_svm(X_train_scaled, y_train, X_val_scaled, y_val)
        models.append(('svm', svm_model))
        model_names.append('SVM')
        
        # Save
        joblib.dump(svm_model, 'models/svm_model_improved.pkl')
        print("Improved SVM saved to: models/svm_model_improved.pkl")
    
    if args.model_type in ['knn', 'all']:
        knn_model = train_improved_knn(X_train_scaled, y_train, X_val_scaled, y_val)
        models.append(('knn', knn_model))
        model_names.append('KNN')
        
        # Save
        joblib.dump(knn_model, 'models/knn_model_improved.pkl')
        print("Improved KNN saved to: models/knn_model_improved.pkl")
    
    # Create ensemble if multiple models
    if args.model_type == 'all' or args.model_type == 'ensemble':
        if len(models) > 1:
            ensemble = create_ensemble(models)
            print("\nTraining ensemble...")
            ensemble.fit(X_train_scaled, y_train)
            
            ensemble_acc = evaluate_model(ensemble, X_test_scaled, y_test, class_map, "Ensemble")
            
            # Save
            joblib.dump(ensemble, 'models/ensemble_model.pkl')
            print("\nEnsemble saved to: models/ensemble_model.pkl")
    
    # Evaluate all models on test set
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    
    results = []
    for i, (name, model) in enumerate(models):
        acc = evaluate_model(model, X_test_scaled, y_test, class_map, model_names[i])
        results.append((model_names[i], acc))
    
    # Summary
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60)
    for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:25s}: {acc*100:.2f}%")
    
    # Save scaler and class map
    joblib.dump(scaler, 'models/scaler_enhanced.pkl')
    with open('models/class_map.json', 'w') as f:
        json.dump(class_map, f, indent=4)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
