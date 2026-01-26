"""
Data loading utilities for PlantVillage dataset.
"""
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from PIL import Image
import numpy as np


class PlantDiseaseDataset:
    """Handles loading and splitting of PlantVillage dataset."""
    
    def __init__(self, data_dir, train_transform=None, val_transform=None):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Path to dataset root (contains class folders)
            train_transform: Transforms for training data
            val_transform: Transforms for validation/test data
        """
        self.data_dir = Path(data_dir)
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        # Check if dataset exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_dir}")
        
        # Get class names
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.num_classes = len(self.classes)
        
        print(f"✓ Found {self.num_classes} disease classes")
        
    def create_datasets(self, train_split=0.7, val_split=0.15, test_split=0.15):
        """
        Create train, validation, and test datasets.
        
        Args:
            train_split: Proportion for training (default 0.7)
            val_split: Proportion for validation (default 0.15)
            test_split: Proportion for testing (default 0.15)
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
        
        # Load full dataset without transforms first
        full_dataset = datasets.ImageFolder(root=str(self.data_dir))
        
        # Calculate split sizes
        dataset_size = len(full_dataset)
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        print(f"\n{'='*50}")
        print(f"Dataset Split Information")
        print(f"{'='*50}")
        print(f"Total images: {dataset_size}")
        print(f"Training:     {train_size} ({train_split*100:.1f}%)")
        print(f"Validation:   {val_size} ({val_split*100:.1f}%)")
        print(f"Testing:      {test_size} ({test_split*100:.1f}%)")
        print(f"{'='*50}\n")
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # Reproducibility
        )
        
        # Apply transforms
        if self.train_transform:
            train_dataset.dataset.transform = self.train_transform
        if self.val_transform:
            val_dataset.dataset.transform = self.val_transform
            test_dataset.dataset.transform = self.val_transform
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset, test_dataset,
                          batch_size=32, num_workers=0):
        """
        Create DataLoaders for train, val, and test sets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size for training (default 32)
            num_workers: Number of worker processes (default 0 for Windows)
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"✓ DataLoaders created")
        print(f"  Batch size: {batch_size}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}\n")
        
        return train_loader, val_loader, test_loader
    
    def get_class_distribution(self):
        """Get number of images per class."""
        distribution = {}
        for class_name in self.classes:
            class_path = self.data_dir / class_name
            num_images = len(list(class_path.glob('*.jpg'))) + \
                        len(list(class_path.glob('*.jpeg'))) + \
                        len(list(class_path.glob('*.png')))
            distribution[class_name] = num_images
        return distribution
    
    def print_class_info(self):
        """Print information about each class."""
        distribution = self.get_class_distribution()
        
        print(f"\n{'='*70}")
        print(f"Class Distribution")
        print(f"{'='*70}")
        print(f"{'Class Name':<40} {'Images':>10}")
        print(f"{'-'*70}")
        
        for class_name, count in sorted(distribution.items()):
            print(f"{class_name:<40} {count:>10}")
        
        print(f"{'-'*70}")
        print(f"{'Total':<40} {sum(distribution.values()):>10}")
        print(f"{'='*70}\n")


class SingleImageDataset(Dataset):
    """Dataset for loading single images for prediction."""
    
    def __init__(self, image_paths, transform=None):
        """
        Initialize single image dataset.
        
        Args:
            image_paths: List of image file paths
            transform: Transform to apply to images
        """
        self.image_paths = image_paths if isinstance(image_paths, list) else [image_paths]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_path


if __name__ == "__main__":
    # Test data loader (will fail if dataset not present)
    print("Testing PlantDiseaseDataset...")
    
    # This will fail if dataset doesn't exist yet - that's expected
    try:
        dataset = PlantDiseaseDataset(
            data_dir="data/raw",
            train_transform=None,
            val_transform=None
        )
        
        dataset.print_class_info()
        
        print("✓ Data loader module ready!")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Dataset not found (expected): {e}")
        print("✓ Data loader module is ready - download dataset to test fully!")