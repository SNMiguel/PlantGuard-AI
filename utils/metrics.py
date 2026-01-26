"""
Evaluation metrics for plant disease classification.
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)


class MetricsCalculator:
    """Calculate and store evaluation metrics."""
    
    def __init__(self, num_classes, class_names=None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: List of class names (optional)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, predictions, labels, probabilities=None):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Model predictions (class indices)
            labels: True labels
            probabilities: Prediction probabilities (optional)
        """
        # Convert to numpy if tensors
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if probabilities is not None and torch.is_tensor(probabilities):
            probabilities = probabilities.cpu().numpy()
        
        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)
        if probabilities is not None:
            self.all_probs.extend(probabilities)
    
    def compute_metrics(self):
        """
        Compute all metrics.
        
        Returns:
            dict: Dictionary containing all metrics
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class
        }
        
        return metrics
    
    def get_confusion_matrix(self):
        """
        Compute confusion matrix.
        
        Returns:
            numpy.ndarray: Confusion matrix
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self):
        """
        Get detailed classification report.
        
        Returns:
            str: Classification report
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        return classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
    
    def print_metrics(self, metrics=None):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Metrics dict (if None, will compute)
        """
        if metrics is None:
            metrics = self.compute_metrics()
        
        print("\n" + "="*60)
        print("Model Performance Metrics")
        print("="*60)
        print(f"Overall Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print("\nMacro-averaged metrics:")
        print(f"  Precision:             {metrics['precision_macro']:.4f}")
        print(f"  Recall:                {metrics['recall_macro']:.4f}")
        print(f"  F1-Score:              {metrics['f1_macro']:.4f}")
        print("\nWeighted-averaged metrics:")
        print(f"  Precision:             {metrics['precision_weighted']:.4f}")
        print(f"  Recall:                {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score:              {metrics['f1_weighted']:.4f}")
        print("="*60 + "\n")
    
    def print_classification_report(self):
        """Print detailed classification report."""
        print("\n" + "="*60)
        print("Detailed Classification Report")
        print("="*60)
        print(self.get_classification_report())
        print("="*60 + "\n")
    
    def get_top_k_accuracy(self, k=5):
        """
        Calculate top-k accuracy.
        
        Args:
            k: Number of top predictions to consider
            
        Returns:
            float: Top-k accuracy
        """
        if not self.all_probs:
            raise ValueError("Probabilities not stored. Pass probabilities to update().")
        
        y_true = np.array(self.all_labels)
        y_probs = np.array(self.all_probs)
        
        # Get top k predictions
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        
        # Check if true label is in top k
        correct = np.any(top_k_preds == y_true[:, np.newaxis], axis=1)
        
        return correct.mean()


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name='metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Update statistics.
        
        Args:
            val: New value
            n: Number of samples (for batch updates)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


if __name__ == "__main__":
    # Test metrics calculator
    print("Testing MetricsCalculator...")
    
    # Generate fake data
    np.random.seed(42)
    num_samples = 100
    num_classes = 5
    
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred = np.random.randint(0, num_classes, num_samples)
    y_probs = np.random.rand(num_samples, num_classes)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # Normalize
    
    # Initialize calculator
    calc = MetricsCalculator(
        num_classes=num_classes,
        class_names=[f"Disease_{i}" for i in range(num_classes)]
    )
    
    # Update with data
    calc.update(y_pred, y_true, y_probs)
    
    # Compute and print metrics
    metrics = calc.compute_metrics()
    calc.print_metrics(metrics)
    
    # Print classification report
    calc.print_classification_report()
    
    # Test top-k accuracy
    top5_acc = calc.get_top_k_accuracy(k=3)
    print(f"Top-3 Accuracy: {top5_acc:.4f}")
    
    print("\n✓ Metrics module ready!")