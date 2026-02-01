"""
Main script for PlantGuard-AI: Plant Disease Classification
Trains ResNet model with transfer learning on PlantVillage dataset.
"""
import torch
import argparse
from pathlib import Path

from models.resnet_model import ResNetClassifier
from utils.data_loader import PlantDiseaseDataset
from utils.transforms import PlantDiseaseTransforms
from utils.metrics import MetricsCalculator
from utils.visualizer import Visualizer
from train import Trainer


def main(args):
    """Main execution function."""
    
    print("\n" + "="*70)
    print(" "*20 + "PlantGuard-AI")
    print(" "*10 + "Plant Disease Classification with PyTorch")
    print("="*70 + "\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Step 1: Setup data transforms
    print("STEP 1: Setting up data transforms")
    print("-" * 70)
    
    transforms = PlantDiseaseTransforms(img_size=224)
    train_transform = transforms.get_train_transforms()
    val_transform = transforms.get_val_transforms()
    
    print("✓ Transforms configured")
    print(f"  Training augmentations: {len(train_transform.transforms)} operations")
    print(f"  Validation transforms: {len(val_transform.transforms)} operations")
    
    # Step 2: Load dataset
    print("\n" + "="*70)
    print("STEP 2: Loading PlantVillage Dataset")
    print("-" * 70)
    
    dataset = PlantDiseaseDataset(
        data_dir=args.data_dir,
        train_transform=train_transform,
        val_transform=val_transform
    )
    
    # Print class information
    if args.verbose:
        dataset.print_class_info()
    
    # Create train/val/test splits
    train_dataset, val_dataset, test_dataset = dataset.create_datasets(
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = dataset.create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Step 3: Initialize model
    print("\n" + "="*70)
    print("STEP 3: Initializing Model")
    print("-" * 70)
    
    model = ResNetClassifier(
        num_classes=dataset.num_classes,
        model_name=args.model,
        pretrained=True
    )
    
    model.summary()
    
    # Optional: Freeze backbone for initial training
    if args.freeze_backbone:
        model.freeze_backbone()
    
    # Step 4: Train model
    print("\n" + "="*70)
    print("STEP 4: Training Model")
    print("-" * 70)
    
    trainer = Trainer(
        model_classifier=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=dataset.num_classes,
        class_names=dataset.classes,
        device=device
    )
    
    history = trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    # Step 5: Evaluate on test set
    print("\n" + "="*70)
    print("STEP 5: Evaluating on Test Set")
    print("-" * 70)
    
    # Load best model
    best_model_path = Path(args.save_dir) / f'{args.model}_best.pth'
    if best_model_path.exists():
        model.load_checkpoint(best_model_path)
        print(f"✓ Loaded best model from {best_model_path}")
    
    # Evaluate
    model.model.eval()
    metrics_calc = MetricsCalculator(dataset.num_classes, dataset.classes)
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model.model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            
            metrics_calc.update(predicted, labels, probabilities)
    
    # Print metrics
    test_metrics = metrics_calc.compute_metrics()
    metrics_calc.print_metrics(test_metrics)
    
    if args.verbose:
        metrics_calc.print_classification_report()
    
    # Step 6: Visualizations
    print("\n" + "="*70)
    print("STEP 6: Generating Visualizations")
    print("-" * 70)
    
    visualizer = Visualizer(save_dir='results/visualizations')
    
    # Confusion matrix
    cm = metrics_calc.get_confusion_matrix()
    visualizer.plot_confusion_matrix(
        cm, dataset.classes, normalize=True,
        save_name=f'{args.model}_confusion_matrix.png'
    )
    
    # Sample predictions
    model.model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images_device = images.to(device)
        outputs = model.model(images_device)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
    
    visualizer.plot_sample_predictions(
        images, labels.cpu().numpy(), predicted.cpu().numpy(),
        dataset.classes, probabilities.cpu().numpy(),
        n_samples=min(16, len(images)),
        save_name=f'{args.model}_sample_predictions.png'
    )
    
    # Final summary
    print("\n" + "="*70)
    print(" "*25 + "SUMMARY")
    print("="*70)
    print(f"Model:              {args.model}")
    print(f"Dataset:            PlantVillage")
    print(f"Classes:            {dataset.num_classes}")
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"Epochs trained:     {args.epochs}")
    print(f"Best val accuracy:  {trainer.best_val_acc:.4f}")
    print(f"Test accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Model saved to:     {args.save_dir}")
    print("="*70 + "\n")
    
    print("✓ Training complete! Check 'results/visualizations' for plots.")
    print(f"✓ Best model saved to: {best_model_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PlantGuard-AI model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone and only train classifier')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='models/saved',
                       help='Directory to save model checkpoints')
    
    # Other arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Run main
    main(args)