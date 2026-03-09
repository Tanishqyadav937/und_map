"""
Dataset and DataLoader classes for Urban Mission Planning Solution.

This module provides custom PyTorch Dataset classes for loading satellite images
and ground truth road masks, with support for train/validation splits and data augmentation.
"""

import os
from typing import Tuple, List, Optional, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RoadSegmentationDataset(Dataset):
    """
    Custom Dataset for loading satellite images and ground truth road masks.
    
    This dataset loads RGB TIFF satellite images and corresponding binary road masks
    for training semantic segmentation models.
    
    Args:
        images_dir: Path to directory containing satellite images (TIFF files)
        masks_dir: Path to directory containing ground truth masks (TIFF files)
        transform: Optional transform to apply to both images and masks
        augment: Whether to apply data augmentation (random flips, rotations)
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        augment: bool = False
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.augment = augment
        
        # Get list of image files
        self.image_files = sorted([
            f for f in os.listdir(images_dir) 
            if f.endswith('.tiff') or f.endswith('.tif')
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No TIFF images found in {images_dir}")
        
        # Verify corresponding masks exist
        self._verify_masks()
    
    def _verify_masks(self):
        """Verify that corresponding mask files exist for all images."""
        for img_file in self.image_files:
            # Mask filename pattern: train_001.tiff -> train_001_map.tiff
            base_name = img_file.replace('.tiff', '').replace('.tif', '')
            mask_file = f"{base_name}_map.tiff"
            mask_path = self.masks_dir / mask_file
            
            if not mask_path.exists():
                # Try alternative naming
                mask_file = f"{base_name}_map.tif"
                mask_path = self.masks_dir / mask_file
                
                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Mask file not found for {img_file}. "
                        f"Expected: {mask_file}"
                    )
    
    def _get_mask_filename(self, image_filename: str) -> str:
        """Get the corresponding mask filename for an image."""
        base_name = image_filename.replace('.tiff', '').replace('.tif', '')
        
        # Try with .tiff extension first
        mask_file = f"{base_name}_map.tiff"
        if (self.masks_dir / mask_file).exists():
            return mask_file
        
        # Try with .tif extension
        mask_file = f"{base_name}_map.tif"
        if (self.masks_dir / mask_file).exists():
            return mask_file
        
        raise FileNotFoundError(f"Mask file not found for {image_filename}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a sample from the dataset.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
            - image_tensor: RGB image as tensor with shape (3, H, W), values in [0, 1]
            - mask_tensor: Binary mask as tensor with shape (1, H, W), values in {0, 1}
        """
        # Load image
        img_file = self.image_files[idx]
        img_path = self.images_dir / img_file
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_file = self._get_mask_filename(img_file)
        mask_path = self.masks_dir / mask_file
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Apply data augmentation if enabled
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        
        # Convert to tensors
        image_tensor = TF.to_tensor(image)  # Converts to [0, 1] range
        mask_tensor = TF.to_tensor(mask)
        
        # Binarize mask (threshold at 0.5)
        mask_tensor = (mask_tensor > 0.5).float()
        
        # Apply custom transform if provided
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, mask_tensor
    
    def _apply_augmentation(
        self, 
        image: Image.Image, 
        mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Apply random data augmentation to image and mask.
        
        Augmentations include:
        - Random horizontal flip (50% probability)
        - Random vertical flip (50% probability)
        - Random rotation (0°, 90°, 180°, or 270°)
        
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (grayscale)
            
        Returns:
            Tuple of augmented (image, mask)
        """
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if torch.rand(1) > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation (0, 90, 180, or 270 degrees)
        angle = torch.randint(0, 4, (1,)).item() * 90
        if angle > 0:
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        return image, mask


def create_dataloaders(
    images_dir: str,
    masks_dir: str,
    batch_size: int = 4,
    train_split: float = 0.8,
    num_workers: int = 0,
    augment_train: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with 80/20 split.
    
    Args:
        images_dir: Path to directory containing satellite images
        masks_dir: Path to directory containing ground truth masks
        batch_size: Batch size for DataLoaders
        train_split: Fraction of data to use for training (default: 0.8)
        num_workers: Number of worker processes for data loading
        augment_train: Whether to apply augmentation to training data
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset without augmentation for splitting
    full_dataset = RoadSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        augment=False
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size],
        generator=generator
    )
    
    # Create separate datasets for train and validation
    # Train dataset with augmentation
    train_dataset = RoadSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        augment=augment_train
    )
    
    # Validation dataset without augmentation
    val_dataset = RoadSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        augment=False
    )
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices.indices)
    val_subset = Subset(val_dataset, val_indices.indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def get_dataset_info(images_dir: str, masks_dir: str) -> dict:
    """
    Get information about the dataset.
    
    Args:
        images_dir: Path to directory containing satellite images
        masks_dir: Path to directory containing ground truth masks
        
    Returns:
        Dictionary with dataset information
    """
    dataset = RoadSegmentationDataset(images_dir, masks_dir)
    
    # Load first sample to get dimensions
    sample_image, sample_mask = dataset[0]
    
    info = {
        'num_samples': len(dataset),
        'image_shape': tuple(sample_image.shape),
        'mask_shape': tuple(sample_mask.shape),
        'image_files': dataset.image_files
    }
    
    return info
