"""
Training script for plant disease classification models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
from pathlib import Path

from models.resnet_model import ResNetClassifier
from utils.data_loader import PlantDiseaseDataset
from utils.transforms import PlantDiseaseTransforms
from utils.metrics import MetricsCalculator, AverageMeter
from utils.visualizer import Visualizer


class Trainer:
    """Handles model training and validation."""
    
    def __init__(self, model_classifier, train_loader, val_loader, 
                 num_classes, class_names, device=None):
        """
        Initialize trainer.
        
        Args:
            model_classifier: Model wrapper (ResNetClassifier or EfficientNetClassifier)
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of classes
            class_names: List of class names
            device: Device to train on
        """
        self.model_classifier = model_classifier
        self.model = model_classifier.get_model()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, 
                                          patience=3, verbose=True)
        
        # Metrics and visualization
        self.visualizer = Visualizer()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        
        print(f"✓ Trainer initialized on {self.device}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        train_loss = AverageMeter('Loss')
        train_acc = AverageMeter('Acc')
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean()
            
            # Update metrics
            train_loss.update(loss.item(), images.size(0))
            train_acc.update(accuracy.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{train_loss.avg:.4f}',
                'acc': f'{train_acc.avg:.4f}'
            })
        
        return train_loss.avg, train_acc.avg
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        val_loss = AverageMeter('Loss')
        val_acc = AverageMeter('Acc')
        
        metrics_calc = MetricsCalculator(self.num_classes, self.class_names)
        
        pbar = tqdm(self.val_loader, desc='Validation')
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == labels).float().mean()
                
                # Update metrics
                val_loss.update(loss.item(), images.size(0))
                val_acc.update(accuracy.item(), images.size(0))
                
                # Store for detailed metrics
                probabilities = torch.softmax(outputs, dim=1)
                metrics_calc.update(predicted, labels, probabilities)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{val_loss.avg:.4f}',
                    'acc': f'{val_acc.avg:.4f}'
                })
        
        return val_loss.avg, val_acc.avg, metrics_calc
    
    def train(self, num_epochs, save_dir='models/saved'):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"Training {self.model_classifier.model_name}")
        print("="*70)
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Device: {self.device}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, metrics_calc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_model_path = save_dir / f'{self.model_classifier.model_name}_best.pth'
                self.model_classifier.save_checkpoint(
                    best_model_path, epoch, self.optimizer,
                    train_loss, val_loss, val_acc
                )
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print("="*70 + "\n")
        
        # Plot training history
        self.visualizer.plot_training_history(
            self.history,
            save_name=f'{self.model_classifier.model_name}_training_history.png'
        )
        
        return self.history


if __name__ == "__main__":
    print("This is the training module. Use main.py to train the complete pipeline.")
    print("✓ Training module ready!")