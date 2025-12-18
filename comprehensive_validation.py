"""
comprehensive_validation.py - Complete Model Evaluation

Evaluates models on train/test split to verify PDF requirements:
- Minimum 0.85 validation accuracy on 6 primary classes
- Comparison between SVM and k-NN
- Performance metrics (precision, recall, F1-score)
- Confusion matrix
"""

import numpy as np
import joblib
import cv2
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

sys.path.insert(0, 'src')
from feature_extraction_enhanced import extract_features


def load_dataset_with_split(dataset_path='dataset', test_size=0.2, random_state=42):
    """
    Load dataset and split into train/test sets
    
    Returns:
        X_train, X_test, y_train, y_test, class_names
    """
    print("\n" + "="*80)
    print("LOADING AND SPLITTING DATASET")
    print("="*80)
    
    dataset_path = Path(dataset_path)
    classes = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
    class_map = {i: cls for i, cls in enumerate(classes)}
    
    X = []
    y = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = dataset_path / class_name
        if not class_path.exists():
            print(f"Warning: {class_path} not found")
            continue
        
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
        print(f"\nProcessing {class_name}: {len(images)} images")
        
        for img_file in tqdm(images, desc=f"  Extracting"):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (128, 128))
            features = extract_features(resized)
            
            X.append(features)
            y.append(class_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Test set is {test_size*100:.0f}% of total data")
    
    return X_train, X_test, y_train, y_test, classes


def evaluate_model(model_name, model_path, scaler_path, X_test, y_test, class_names):
    """
    Evaluate a trained model on test set
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*80)
    print(f"EVALUATING {model_name.upper()}")
    print("="*80)
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'name': model_name,
        'accuracy': accuracy,
        'predictions': y_pred,
        'confusion_matrix': cm
    }


def plot_confusion_matrices(results, class_names, save_path='validation_results'):
    """Plot confusion matrices for all models"""
    Path(save_path).mkdir(exist_ok=True)
    
    for result in results:
        plt.figure(figsize=(10, 8))
        cm = result['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'{result["name"]} - Confusion Matrix (Normalized)\nAccuracy: {result["accuracy"]:.4f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = f'{save_path}/{result["name"].lower().replace(" ", "_")}_confusion_matrix.png'
        plt.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
        plt.close()


def generate_validation_report(results, save_path='validation_results'):
    """Generate comprehensive validation report"""
    Path(save_path).mkdir(exist_ok=True)
    
    report_file = f'{save_path}/validation_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE MODEL VALIDATION REPORT\n")
        f.write("Material Stream Identification System\n")
        f.write("="*80 + "\n\n")
        
        f.write("PDF REQUIREMENT: Minimum 0.85 (85%) validation accuracy\n")
        f.write("-"*80 + "\n\n")
        
        # Results summary
        f.write("MODEL PERFORMANCE SUMMARY:\n")
        f.write("-"*80 + "\n")
        for result in results:
            accuracy = result['accuracy']
            status = "✅ MEETS REQUIREMENT" if accuracy >= 0.85 else "⚠️ BELOW TARGET"
            f.write(f"{result['name']:20s}: {accuracy:.4f} ({accuracy*100:.2f}%) {status}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("BEST MODEL: ")
        best_model = max(results, key=lambda x: x['accuracy'])
        f.write(f"{best_model['name']} with {best_model['accuracy']:.4f} accuracy\n")
        f.write("="*80 + "\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-"*80 + "\n")
        if best_model['accuracy'] < 0.85:
            f.write("⚠️ Current accuracy is below the 0.85 target.\n")
            f.write("Possible improvements:\n")
            f.write("  1. Add more diverse training data\n")
            f.write("  2. Try additional feature engineering\n")
            f.write("  3. Ensemble multiple models\n")
            f.write("  4. Fine-tune hyperparameters further\n")
        else:
            f.write("✅ Model meets the PDF requirement of 0.85 validation accuracy!\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\nValidation report saved to: {report_file}")
    
    return report_file


def main():
    """Main validation workflow"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("PDF Requirement: Minimum 0.85 validation accuracy on 6 primary classes")
    print("="*80)
    
    # Load and split dataset
    X_train, X_test, y_train, y_test, class_names = load_dataset_with_split(
        dataset_path='dataset',
        test_size=0.2,
        random_state=42
    )
    
    # Evaluate all models
    results = []
    
    models_to_test = [
        ('SVM Enhanced', 'models/svm_enhanced.pkl'),
        ('k-NN Enhanced', 'models/knn_enhanced.pkl'),
        ('Ensemble Enhanced', 'models/ensemble_enhanced.pkl')
    ]
    
    for model_name, model_path in models_to_test:
        if Path(model_path).exists():
            result = evaluate_model(
                model_name, model_path, 'models/scaler_enhanced.pkl',
                X_test, y_test, class_names
            )
            results.append(result)
        else:
            print(f"\nWarning: {model_path} not found, skipping...")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    plot_confusion_matrices(results, class_names)
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING VALIDATION REPORT")
    print("="*80)
    generate_validation_report(results)
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\nResults:")
    for result in results:
        accuracy = result['accuracy']
        status = "[PASS]" if accuracy >= 0.85 else "[WARN]"
        print(f"  {status} {result['name']:20s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n[BEST] Best Model: {best['name']} ({best['accuracy']*100:.2f}%)")
    
    if best['accuracy'] >= 0.85:
        print("\n[PASS] PDF REQUIREMENT MET: Validation accuracy >= 0.85")
    else:
        print(f"\n[WARN] PDF TARGET NOT MET: {best['accuracy']:.4f} < 0.85")
        print("   (Current balanced dataset with 3000 images achieves ~78% accuracy)")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
