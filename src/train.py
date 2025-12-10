import os
import json
import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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
    return feats, labels, class_map


def scale_features(X_train, X_val, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return scaler, X_train_scaled, X_val_scaled, X_test_scaled
    return scaler, X_train_scaled, X_val_scaled


def train_svm_grid(X_train, y_train, cv=3, n_jobs=-1, verbose=1):
    """
    Grid search over SVM hyperparameters.
    Choices justification:
      - kernel: ['linear', 'rbf'] -> test linear separability vs non-linear.
      - C: [0.1, 1, 5, 10] -> regularization strength sweep.
      - gamma: ['scale', 'auto'] for RBF; we rely on 'scale' mostly.
    """
    svc = SVC(probability=True, random_state=RANDOM_STATE)
    param_grid = {
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1, 5, 10],
        # 'gamma' only used by RBF; GridSearchCV will ignore for linear kernel automatically
        'gamma': ['scale', 'auto']
    }
    grid = GridSearchCV(svc, param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    return best, grid.best_params_, grid


def train_knn_grid(X_train, y_train, cv=3):
    """
    Simple hyperparameter search for k-NN.
    We try multiple k and both weight schemes:
      - 'uniform' : every neighbor votes equally
      - 'distance': closer neighbors have larger influence (useful for noisy feature spaces)
    We pick the best model by cross-validated accuracy on the training fold.
    """
    best_model = None
    best_score = -1.0
    best_params = None

    for k in [3, 5, 7, 9]:
        for weights in ['uniform', 'distance']:
            knn = KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=-1)
            # use cross-validation to get a robust estimate
            scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_model = KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=-1)
                best_model.fit(X_train, y_train)
                best_params = {'n_neighbors': k, 'weights': weights, 'cv_score': mean_score}
    return best_model, best_params


def evaluate_and_report(model, X, y, class_map=None):
    preds = model.predict(X)
    acc = float(accuracy_score(y, preds))
    report = classification_report(y, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, preds)
    return acc, report, cm


def main(args):
    np.random.seed(RANDOM_STATE)

    feats, labels, class_map = load_features(args.features)
    print(f'Loaded features: X.shape={feats.shape}, y.shape={labels.shape}, classes={len(np.unique(labels))}')

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        feats, labels, test_size=args.test_size, random_state=RANDOM_STATE, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.val_size, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Scale features (important for SVM)
    scaler, X_train_s, X_val_s, X_test_s = None, None, None, None
    scaler, X_train_s, X_val_s, X_test_s = scale_features(X_train, X_val, X_test=X_test)

    os.makedirs(args.model_dir, exist_ok=True)
    scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f'Scaler saved -> {scaler_path}')

    # Train SVM (grid search)
    print('Training SVM (GridSearchCV)...')
    svm_model, svm_best_params, svm_grid = train_svm_grid(X_train_s, y_train, cv=args.cv)
    svm_path = os.path.join(args.model_dir, 'svm_model.pkl')
    joblib.dump(svm_model, svm_path)
    print(f'SVM saved -> {svm_path}')
    print('SVM best params:', svm_best_params)

    # Train k-NN (grid search via CV)
    print('Training k-NN (CV search)...')
    knn_model, knn_best_params = train_knn_grid(X_train_s, y_train, cv=args.cv)
    knn_path = os.path.join(args.model_dir, 'knn_model.pkl')
    joblib.dump(knn_model, knn_path)
    print(f'k-NN saved -> {knn_path}')
    print('k-NN best params:', knn_best_params)

    # Evaluate both on validation set
    print('Evaluating on validation set...')
    svm_val_acc, svm_val_report, svm_val_cm = evaluate_and_report(svm_model, X_val_s, y_val, class_map)
    knn_val_acc, knn_val_report, knn_val_cm = evaluate_and_report(knn_model, X_val_s, y_val, class_map)
    print(f'SVM val acc: {svm_val_acc:.4f}  |  k-NN val acc: {knn_val_acc:.4f}')

    # Select best model by validation accuracy (primary), fallback to SVM on tie
    if svm_val_acc >= knn_val_acc:
        best_model = svm_model
        best_name = 'svm'
    else:
        best_model = knn_model
        best_name = 'knn'

    best_path = os.path.join(args.model_dir, 'best_model.pkl')
    joblib.dump(best_model, best_path)
    print(f'Best model: {best_name} saved -> {best_path}')

    # Final evaluation on held-out test set
    print('Evaluating selected best model on held-out test set...')
    best_test_acc, best_test_report, best_test_cm = evaluate_and_report(best_model, X_test_s, y_test, class_map)
    print(f'Best model test accuracy: {best_test_acc:.4f}')

    # Save training report summary
    report = {
        'svm': {
            'best_params': svm_best_params,
            'val_accuracy': svm_val_acc,
        },
        'knn': {
            'best_params': knn_best_params,
            'val_accuracy': knn_val_acc,
        },
        'selected': {
            'model': best_name,
            'test_accuracy': best_test_acc,
        },
        'class_map': class_map
    }
    report_path = os.path.join(args.model_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Training report saved -> {report_path}')

    # Print short classification report to console
    print('--- Best Model Test Classification Report ---')
    print(json.dumps(best_test_report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SVM and k-NN on feature vectors')
    parser.add_argument('--features', type=str, required=True, help='Path to .npz features file (feats, labels, class_map)')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save models and scaler')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation fraction of training data (after initial split)')
    parser.add_argument('--test_size', type=float, default=0.1, help='Held-out test fraction from full data')
    parser.add_argument('--cv', type=int, default=3, help='Cross-validation folds for hyperparameter search')
    args = parser.parse_args()

    main(args)