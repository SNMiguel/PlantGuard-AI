"""
Visualization utilities for training progress and results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
import itertools

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class Visualizer:
    """Handle all visualization tasks."""
    
    def __init__(self, save_dir='results/visualizations'):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self, history, save_name='training_history.png'):
        """
        Plot training and validation loss/accuracy curves.
        
        Args:
            history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
            save_name: Filename to save plot
        """
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm, class_names, normalize=False, 
                             save_name='confusion_matrix.png'):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize the matrix
            save_name: Filename to save plot
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.5), 
                                        max(10, len(class_names) * 0.5)))
        
        # Plot heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set ticks
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=title,
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=8)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_sample_predictions(self, images, true_labels, pred_labels, 
                               class_names, probs=None, n_samples=16,
                               save_name='sample_predictions.png'):
        """
        Plot grid of sample predictions.
        
        Args:
            images: Batch of images (tensors)
            true_labels: True class indices
            pred_labels: Predicted class indices
            class_names: List of class names
            probs: Prediction probabilities (optional)
            n_samples: Number of samples to show
            save_name: Filename to save plot
        """
        n_samples = min(n_samples, len(images))
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        for idx in range(n_samples):
            ax = axes[idx]
            
            # Denormalize image
            img = images[idx].cpu()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            
            # Display image
            ax.imshow(img)
            
            # Create title
            true_class = class_names[true_labels[idx]]
            pred_class = class_names[pred_labels[idx]]
            
            if probs is not None:
                prob = probs[idx][pred_labels[idx]]
                title = f"True: {true_class}\nPred: {pred_class} ({prob:.2%})"
            else:
                title = f"True: {true_class}\nPred: {pred_class}"
            
            # Color title based on correctness
            color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
        
        # Hide extra subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sample predictions saved to {save_path}")
        plt.close()
    
    def plot_class_distribution(self, class_counts, class_names, 
                               save_name='class_distribution.png'):
        """
        Plot class distribution bar chart.
        
        Args:
            class_counts: List/array of counts per class
            class_names: List of class names
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.3), 6))
        
        bars = ax.bar(range(len(class_counts)), class_counts, color='steelblue', alpha=0.8)
        
        ax.set_xlabel('Disease Class', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Class distribution saved to {save_path}")
        plt.close()
    
    def plot_top_k_accuracy(self, k_values, accuracies, 
                           save_name='top_k_accuracy.png'):
        """
        Plot top-k accuracy curve.
        
        Args:
            k_values: List of k values
            accuracies: Corresponding accuracy values
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(k_values, accuracies, 'b-o', linewidth=2, markersize=8)
        ax.set_xlabel('k (Top-k Predictions)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Top-k Accuracy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Top-k accuracy plot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    # Test visualizer
    print("Testing Visualizer...")
    
    viz = Visualizer(save_dir='results/test_viz')
    
    # Test training history plot
    history = {
        'train_loss': [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6],
        'val_loss': [2.3, 1.9, 1.6, 1.3, 1.1, 1.0, 0.9, 0.85],
        'train_acc': [0.3, 0.45, 0.6, 0.7, 0.75, 0.82, 0.87, 0.9],
        'val_acc': [0.35, 0.48, 0.58, 0.68, 0.72, 0.78, 0.82, 0.85]
    }
    viz.plot_training_history(history)
    
    # Test confusion matrix
    cm = np.random.randint(0, 100, (5, 5))
    class_names = [f"Disease_{i}" for i in range(5)]
    viz.plot_confusion_matrix(cm, class_names, normalize=True)
    
    print("\n✓ Visualizer module ready!")