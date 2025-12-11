"""
evaluate.py
Standalone evaluation script for trained models.
Loads a saved model and evaluates it on a test dataset or feature file.
"""

import argparse
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_features(npz_path):
    """Load features from npz file."""
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


def evaluate_model(model, X, y, dataset_name="Test", class_map=None):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained sklearn model
        X: Feature matrix
        y: True labels
        dataset_name: Name of the dataset being evaluated
        class_map: Dictionary mapping label indices to class names
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name} Set")
    print(f"{'='*60}")
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    # Print overall metrics
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification report
    target_names = None
    if class_map:
        target_names = [class_map[i] for i in sorted(class_map.keys())]
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y, y_pred, target_names=target_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Prepare results dictionary
    results = {
        'dataset': dataset_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    }
    
    return results, cm, target_names


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_metrics(report_dict, class_names, save_path=None):
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        report_dict: Classification report dictionary
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    metrics = ['precision', 'recall', 'f1-score']
    
    # Extract per-class metrics
    data = {metric: [] for metric in metrics}
    
    for class_name in class_names:
        if class_name in report_dict:
            for metric in metrics:
                data[metric].append(report_dict[class_name][metric])
    
    # Create plot
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, data['precision'], width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, data['recall'], width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, data['f1-score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test data')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to saved model (.pkl file)')
    parser.add_argument('--scaler', type=str, default=None,
                        help='Path to saved scaler (.pkl file, optional)')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to features file (.npz)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save evaluation plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = joblib.load(args.model)
    print(f"Model loaded successfully: {type(model).__name__}")
    
    # Load scaler if provided
    scaler = None
    if args.scaler:
        print(f"Loading scaler from: {args.scaler}")
        scaler = joblib.load(args.scaler)
        print("Scaler loaded successfully")
    
    # Load features
    print(f"\nLoading features from: {args.features}")
    X, y, class_map = load_features(args.features)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Apply scaling if scaler is provided
    if scaler:
        print("\nApplying feature scaling...")
        X_scaled = scaler.transform(X)
    else:
        print("\nNo scaler provided, using raw features")
        X_scaled = X
    
    # Evaluate model
    results, cm, class_names = evaluate_model(model, X_scaled, y, "Test", class_map)
    
    # Save results to JSON
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to: {results_path}")
    
    # Generate and save plots if requested
    if args.save_plots:
        print("\nGenerating evaluation plots...")
        
        # Confusion matrix plot
        cm_plot_path = output_dir / 'confusion_matrix.png'
        plot_confusion_matrix(cm, class_names, save_path=cm_plot_path)
        
        # Per-class metrics plot
        metrics_plot_path = output_dir / 'per_class_metrics.png'
        plot_per_class_metrics(
            results['classification_report'], 
            class_names, 
            save_path=metrics_plot_path
        )
        
        print("\nAll plots saved successfully!")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
