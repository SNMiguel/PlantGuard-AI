"""
ResNet model with transfer learning for plant disease classification.
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


class ResNetClassifier:
    """ResNet-based classifier with transfer learning."""
    
    def __init__(self, num_classes, model_name='resnet18', pretrained=True):
        """
        Initialize ResNet classifier.
        
        Args:
            num_classes: Number of output classes
            model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50')
            pretrained: Use ImageNet pretrained weights
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained ResNet
        print(f"Loading {model_name} (pretrained={pretrained})...")
        
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Replace final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        print(f"✓ Model initialized on {self.device}")
        print(f"  Total parameters: {self.count_parameters():,}")
        print(f"  Trainable parameters: {self.count_parameters(trainable_only=True):,}")
    
    def count_parameters(self, trainable_only=False):
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())
    
    def freeze_backbone(self):
        """Freeze all layers except the final FC layer."""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        
        print("✓ Backbone frozen - only training final layer")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        print("✓ Backbone unfrozen - fine-tuning all layers")
    
    def get_model(self):
        """Return the PyTorch model."""
        return self.model
    
    def save_checkpoint(self, filepath, epoch, optimizer, train_loss, val_loss, val_acc):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state
            train_loss: Training loss
            val_loss: Validation loss
            val_acc: Validation accuracy
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath, optimizer=None):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            optimizer: Optimizer to load state into (optional)
            
        Returns:
            dict: Checkpoint information
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        print(f"Loading checkpoint from {filepath}...")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Checkpoint loaded (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f})")
        
        return checkpoint
    
    def predict(self, images):
        """
        Make predictions on a batch of images.
        
        Args:
            images: Batch of images (tensor)
            
        Returns:
            tuple: (predicted_classes, probabilities)
        """
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities
    
    def get_feature_maps(self, images, layer_name='layer4'):
        """
        Extract feature maps from intermediate layer.
        
        Args:
            images: Batch of images (tensor)
            layer_name: Name of layer to extract from
            
        Returns:
            tensor: Feature maps
        """
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # Register hook
        layer = dict(self.model.named_modules())[layer_name]
        handle = layer.register_forward_hook(hook_fn)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            _ = self.model(images)
        
        # Remove hook
        handle.remove()
        
        return features[0]
    
    def summary(self):
        """Print model summary."""
        print("\n" + "="*70)
        print(f"{self.model_name.upper()} Model Summary")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Total parameters: {self.count_parameters():,}")
        print(f"Trainable parameters: {self.count_parameters(trainable_only=True):,}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Test ResNet model
    print("Testing ResNetClassifier...")
    
    # Initialize model
    classifier = ResNetClassifier(
        num_classes=38,  # PlantVillage has 38 classes
        model_name='resnet18',
        pretrained=True
    )
    
    # Print summary
    classifier.summary()
    
    # Test with dummy input
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    predictions, probabilities = classifier.predict(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
    # Test freezing/unfreezing
    classifier.freeze_backbone()
    print(f"Trainable params (frozen): {classifier.count_parameters(trainable_only=True):,}")
    
    classifier.unfreeze_backbone()
    print(f"Trainable params (unfrozen): {classifier.count_parameters(trainable_only=True):,}")
    
    print("\n✓ ResNet model ready!")