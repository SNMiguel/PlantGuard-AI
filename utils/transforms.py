"""
Data augmentation and preprocessing transforms for plant disease images.
"""
import torch
from torchvision import transforms
from PIL import Image


class PlantDiseaseTransforms:
    """Handles image transformations for training and inference."""
    
    def __init__(self, img_size=224):
        """
        Initialize transforms.
        
        Args:
            img_size (int): Target image size (default 224 for ResNet/EfficientNet)
        """
        self.img_size = img_size
        
        # ImageNet normalization (pre-trained models expect this)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def get_train_transforms(self):
        """
        Get augmentation pipeline for training data.
        Includes random flips, rotations, and color jittering.
        """
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def get_val_transforms(self):
        """
        Get transforms for validation/test data.
        Only resizing and normalization (no augmentation).
        """
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def get_test_transforms(self):
        """Alias for validation transforms."""
        return self.get_val_transforms()
    
    @staticmethod
    def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Denormalize tensor for visualization.
        
        Args:
            tensor: Normalized image tensor
            mean: Mean used for normalization
            std: Std used for normalization
            
        Returns:
            Denormalized tensor
        """
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        return tensor * std + mean
    
    @staticmethod
    def load_and_transform_image(image_path, transform):
        """
        Load image from path and apply transform.
        
        Args:
            image_path: Path to image file
            transform: Transform to apply
            
        Returns:
            Transformed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        return transform(image)


if __name__ == "__main__":
    # Test transforms
    print("Testing PlantDiseaseTransforms...")
    
    transformer = PlantDiseaseTransforms(img_size=224)
    
    train_transform = transformer.get_train_transforms()
    val_transform = transformer.get_val_transforms()
    
    print(f"✓ Train transforms: {len(train_transform.transforms)} operations")
    print(f"✓ Validation transforms: {len(val_transform.transforms)} operations")
    print("\nTrain transform pipeline:")
    for i, t in enumerate(train_transform.transforms, 1):
        print(f"  {i}. {t.__class__.__name__}")
    
    print("\n✓ Transforms module ready!")