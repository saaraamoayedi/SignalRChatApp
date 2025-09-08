"""
Dataset classes for medical image segmentation with advanced augmentations.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class SegmentationDataset(Dataset):
    """Advanced segmentation dataset with proper augmentations."""
    
    def __init__(self, image_dir, mask_dir, image_size=(384, 384), 
                 transform=None, augment=False, normalize=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform
        self.augment = augment
        self.normalize = normalize
        
        # Get sorted file lists
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        assert len(self.images) == len(self.masks), \
            f"Number of images ({len(self.images)}) != number of masks ({len(self.masks)})"
        
        # Advanced augmentation pipeline
        if self.augment:
            self.aug_transform = A.Compose([
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                
                # Elastic transformations
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(p=0.2),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),
                
                # Intensity augmentations
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
                
                # Noise and blur
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.1),
                A.MotionBlur(blur_limit=3, p=0.1),
                
                # Advanced augmentations
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
                
                # Resize to target size
                A.Resize(height=image_size[0], width=image_size[1]),
            ], additional_targets={'mask': 'mask'})
        else:
            self.basic_transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
            ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        # Load as RGB and grayscale
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply augmentations
        if self.augment:
            augmented = self.aug_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            transformed = self.basic_transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to float32 and normalize
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        
        if self.normalize:
            image = image / 255.0
        
        # Normalize mask to [0, 1]
        if mask.max() > 1:
            mask = mask / 255.0
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        # Apply additional transforms if provided
        if self.transform:
            image = self.transform(image)
            
        return image, mask


class SimpleSegmentationDataset(Dataset):
    """Simple segmentation dataset compatible with your original code."""
    
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(384, 384)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB").resize(self.image_size)
        mask = Image.open(mask_path).convert("L").resize(self.image_size)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)

        mask = torch.from_numpy(mask).float()
        return image, mask


def create_data_loaders(train_image_path, train_mask_path, 
                       test_image_path, test_mask_path,
                       batch_size=8, image_size=(384, 384), 
                       num_workers=4, use_advanced_augmentation=True):
    """Create training and validation data loaders."""
    
    if use_advanced_augmentation:
        train_dataset = SegmentationDataset(
            train_image_path, train_mask_path, 
            image_size=image_size, augment=True
        )
        test_dataset = SegmentationDataset(
            test_image_path, test_mask_path, 
            image_size=image_size, augment=False
        )
    else:
        # Use simple transforms like your original code
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = SimpleSegmentationDataset(
            train_image_path, train_mask_path, 
            transform=transform, image_size=image_size
        )
        test_dataset = SimpleSegmentationDataset(
            test_image_path, test_mask_path, 
            transform=transform, image_size=image_size
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def mix_augment(x, y, alpha=0.4, prob_mixup=0.33, prob_cutmix=0.33, prob_alphamix=0.34):
    """
    Apply MixUp + CutMix + AlphaMix augmentations randomly.
    
    Args:
        x: batch images  (B, C, H, W)
        y: batch masks   (B, 1, H, W)
        alpha: Beta distribution parameter
        prob_mixup: Probability of applying MixUp
        prob_cutmix: Probability of applying CutMix
        prob_alphamix: Probability of applying AlphaMix
    
    Returns:
        Augmented images and masks
    """
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)

    # Sample λ from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Decide which technique to apply
    r = random.random()

    # ----------- MixUp -----------
    if r < prob_mixup:
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]

    # ----------- CutMix -----------
    elif r < prob_mixup + prob_cutmix:
        # Select patch coordinates
        cut_w = int(W * np.sqrt(1 - lam))
        cut_h = int(H * np.sqrt(1 - lam))
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1, y1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
        x2, y2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)

        mixed_x = x.clone()
        mixed_y = y.clone()
        mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        mixed_y[:, :, y1:y2, x1:x2] = y[index, :, y1:y2, x1:x2]

    # ----------- AlphaMix -----------
    else:
        # Use mask as mixing coefficient
        mask_lambda = torch.rand_like(y)  # Random noise
        lam_mask = y * mask_lambda + (1 - y) * (1 - mask_lambda)

        mixed_x = lam_mask * x + (1 - lam_mask) * x[index, :]
        mixed_y = lam_mask * y + (1 - lam_mask) * y[index, :]

    return mixed_x, mixed_y


__all__ = ["SegmentationDataset", "SimpleSegmentationDataset", "create_data_loaders", "mix_augment"]