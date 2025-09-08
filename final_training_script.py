import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import ttach as tta
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Import advanced techniques
from advanced_techniques import (
    WarmupCosineScheduler, SAM, MixUp, CutMix, PuzzleMix,
    GradientAccumulator, MultiScaleWrapper, LabelSmoothingBCE
)

# Check GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ GPU is available!")
    print("Device name:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
    print("⚠️ GPU not detected. Running on CPU.")

# Paths
train_image_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/train/images'
train_mask_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/train/masks'
test_image_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/test/images'
test_mask_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/test/masks'

# Enhanced Configuration
class Config:
    # Data
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    BATCH_SIZE = 6  # Reduced for larger images and gradient accumulation
    NUM_WORKERS = 4
    
    # Model
    ENCODER = "timm-efficientnet-b5"  # Better encoder
    NUM_CLASSES = 1
    
    # Training
    EPOCHS = 250
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    GRADIENT_ACCUMULATION = 2
    
    # Augmentation probabilities
    MIXUP_PROB = 0.2
    CUTMIX_PROB = 0.2
    PUZZLE_PROB = 0.1
    
    # Advanced training
    USE_SAM = True
    SAM_RHO = 0.05
    WARMUP_EPOCHS = 10
    LABEL_SMOOTHING = 0.05
    
    # K-Fold
    N_FOLDS = 5
    
    # Post-processing
    TTA_TRANSFORMS = True
    MIN_OBJECT_SIZE = 150
    MORPHOLOGY_KERNEL = 5

config = Config()

# Enhanced Augmentations
def get_heavy_augmentation():
    return A.Compose([
        A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Geometric
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
        ], p=0.5),
        
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=45, 
                          border_mode=cv2.BORDER_REFLECT, p=0.5),
        
        # Pixel-level
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        ], p=0.8),
        
        # Noise & Blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=7, p=1),
            A.MedianBlur(blur_limit=7, p=1),
        ], p=0.3),
        
        # Dropout
        A.OneOf([
            A.CoarseDropout(max_holes=8, max_height=64, max_width=64, min_holes=1,
                          min_height=16, min_width=16, fill_value=0, mask_fill_value=0, p=1),
            A.MaskDropout(max_objects=3, p=1),
        ], p=0.3),
        
        # Advanced
        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
                          num_shadows_upper=2, shadow_dimension=5, p=1),
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.8, alpha_coef=0.08, p=1),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, 
                        drop_width=1, drop_color=(200, 200, 200), p=1),
        ], p=0.1),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# Enhanced Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, use_cache=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.use_cache = use_cache
        self.cache = {}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read with OpenCV for better performance
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0).float() / 255.0
        
        if self.use_cache and len(self.cache) < 1000:  # Limit cache size
            self.cache[idx] = (image, mask)
        
        return image, mask

# Enhanced Model with FPN
class EnhancedSegmentationModel(nn.Module):
    def __init__(self, encoder_name=config.ENCODER, num_classes=config.NUM_CLASSES):
        super().__init__()
        
        # Main model - FPN for better multi-scale features
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
            decoder_dropout=0.2,
            decoder_merge_policy="cat",  # Concatenate features
        )
        
        # Additional refinement head
        self.refinement = nn.Sequential(
            nn.Conv2d(num_classes, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
        
    def forward(self, x):
        out = self.model(x)
        out = out + self.refinement(out)  # Residual connection
        return out

# Hybrid Loss
class HybridLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        self.label_smoothing_bce = LabelSmoothingBCE(smoothing=config.LABEL_SMOOTHING)
        
    def forward(self, logits, targets, use_label_smoothing=True):
        # Dice Loss
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        intersection = (probs * targets).sum(dims)
        cardinality = (probs + targets).sum(dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = (1. - dice_score).mean()
        
        # Focal Tversky Loss
        alpha, beta = 0.7, 0.3
        tp = (probs * targets).sum(dims)
        fp = (probs * (1 - targets)).sum(dims)
        fn = ((1 - probs) * targets).sum(dims)
        tversky = (tp + self.smooth) / (tp + alpha * fp + beta * fn + self.smooth)
        focal_tversky = ((1 - tversky) ** 0.75).mean()
        
        # BCE or Label Smoothing BCE
        if use_label_smoothing:
            bce_loss = self.label_smoothing_bce(logits, targets)
        else:
            bce_loss = self.bce(logits, targets)
        
        # Boundary Loss (simplified)
        laplacian_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                      dtype=torch.float32).to(logits.device)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        
        boundary_targets = F.conv2d(targets, laplacian_kernel, padding=1)
        boundary_preds = F.conv2d(probs, laplacian_kernel, padding=1)
        boundary_loss = F.mse_loss(boundary_preds, boundary_targets)
        
        # Weighted combination
        total_loss = (0.3 * bce_loss + 
                     0.3 * dice_loss + 
                     0.3 * focal_tversky + 
                     0.1 * boundary_loss)
        
        return total_loss

# IoU metric
def calculate_iou(pred, target, threshold=0.5, smooth=1e-6):
    pred = (pred > threshold).float()
    intersection = torch.sum(pred * target, dim=[1, 2, 3])
    union = torch.sum(pred + target, dim=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

# Advanced post-processing
def advanced_post_process(pred_mask, min_size=150, kernel_size=5):
    """Enhanced post-processing with multiple techniques"""
    # Convert to numpy
    mask = pred_mask.cpu().numpy().astype(np.uint8)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Fill holes
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small objects
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
    mask_clean = np.zeros_like(mask_filled)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask_clean[labels == i] = 1
    
    # Smooth boundaries
    mask_smooth = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_smooth = cv2.GaussianBlur(mask_smooth.astype(np.float32), (5, 5), 0)
    mask_smooth = (mask_smooth > 0.5).astype(np.uint8)
    
    return torch.from_numpy(mask_smooth).float()

# Training function with SAM
def train_epoch_sam(model, loader, criterion, optimizer, scaler, device, 
                   accumulator, mixup, cutmix, puzzle_mix):
    model.train()
    total_loss = 0
    total_iou = 0
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Apply mix augmentations
        r = np.random.random()
        if r < config.MIXUP_PROB:
            images, masks = mixup(images, masks)
        elif r < config.MIXUP_PROB + config.CUTMIX_PROB:
            images, masks = cutmix(images, masks)
        elif r < config.MIXUP_PROB + config.CUTMIX_PROB + config.PUZZLE_PROB:
            images, masks = puzzle_mix(images, masks)
        
        # First forward-backward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = accumulator.scale_loss(loss)
        
        scaler.scale(loss).backward()
        
        if config.USE_SAM and accumulator.should_step():
            # SAM first step
            optimizer.first_step(zero_grad=True)
            
            # Second forward-backward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = accumulator.scale_loss(loss)
            
            scaler.scale(loss).backward()
            optimizer.second_step(zero_grad=True)
        elif accumulator.should_step():
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            iou = calculate_iou(preds, masks)
        
        total_loss += loss.item() * accumulator.accumulation_steps
        total_iou += iou.item()
    
    return total_loss / len(loader), total_iou / len(loader)

# Validation with TTA
def validate_with_tta(model, loader, criterion, device):
    # TTA transforms
    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ])
    
    tta_model = tta.SegmentationTTAWrapper(model, tta_transforms, merge_mode='mean')
    
    model.eval()
    total_loss = 0
    total_iou = 0
    total_iou_pp = 0  # With post-processing
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # TTA predictions
            outputs = tta_model(images)
            loss = criterion(outputs, masks, use_label_smoothing=False)
            
            preds = torch.sigmoid(outputs)
            
            # Regular IoU
            iou = calculate_iou(preds, masks)
            
            # IoU with post-processing
            preds_pp = torch.stack([
                advanced_post_process(preds[i, 0], config.MIN_OBJECT_SIZE, config.MORPHOLOGY_KERNEL)
                for i in range(preds.shape[0])
            ]).unsqueeze(1).to(device)
            iou_pp = calculate_iou(preds_pp, masks)
            
            total_loss += loss.item()
            total_iou += iou.item()
            total_iou_pp += iou_pp.item()
    
    return (total_loss / len(loader), 
            total_iou / len(loader), 
            total_iou_pp / len(loader))

# K-Fold training
def train_kfold():
    # Load all file paths
    all_images = sorted([os.path.join(train_image_path, f) for f in os.listdir(train_image_path)])
    all_masks = sorted([os.path.join(train_mask_path, f) for f in os.listdir(train_mask_path)])
    
    kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_images)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{config.N_FOLDS}")
        print(f"{'='*50}")
        
        # Split data
        train_images = [all_images[i] for i in train_idx]
        train_masks = [all_masks[i] for i in train_idx]
        val_images = [all_images[i] for i in val_idx]
        val_masks = [all_masks[i] for i in val_idx]
        
        # Create datasets
        train_dataset = SegmentationDataset(
            train_images, train_masks,
            transform=get_heavy_augmentation()
        )
        val_dataset = SegmentationDataset(
            val_images, val_masks,
            transform=get_validation_augmentation()
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=True
        )
        
        # Initialize model
        model = EnhancedSegmentationModel().to(device)
        model = MultiScaleWrapper(model, scales=[0.75, 1.0, 1.25])
        
        # Loss and optimizer
        criterion = HybridLoss()
        
        if config.USE_SAM:
            base_optimizer = lambda params, **kwargs: optim.AdamW(params, **kwargs)
            optimizer = SAM(model.parameters(), base_optimizer, 
                          rho=config.SAM_RHO, lr=config.LEARNING_RATE, 
                          weight_decay=config.WEIGHT_DECAY)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
        
        # Scheduler
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=config.WARMUP_EPOCHS,
            total_epochs=config.EPOCHS, min_lr=1e-6
        )
        
        # Training utilities
        scaler = GradScaler()
        accumulator = GradientAccumulator(config.GRADIENT_ACCUMULATION)
        mixup = MixUp(alpha=1.0)
        cutmix = CutMix(alpha=1.0)
        puzzle_mix = PuzzleMix(alpha=1.0, beta=1.0)
        
        # Training loop
        best_iou = 0
        patience = 30
        patience_counter = 0
        
        for epoch in range(config.EPOCHS):
            # Training
            if config.USE_SAM:
                train_loss, train_iou = train_epoch_sam(
                    model, train_loader, criterion, optimizer, scaler, device,
                    accumulator, mixup, cutmix, puzzle_mix
                )
            else:
                # Regular training (implement if needed)
                pass
            
            # Validation with TTA
            val_loss, val_iou, val_iou_pp = validate_with_tta(
                model.module if hasattr(model, 'module') else model, 
                val_loader, criterion, device
            )
            
            # Update scheduler
            scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1}/{config.EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val IoU (PP): {val_iou_pp:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_iou_pp > best_iou:
                best_iou = val_iou_pp
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'epoch': epoch,
                    'fold': fold
                }, f'best_model_fold{fold}.pth')
                print(f"✅ New best IoU: {best_iou:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
        
        fold_results.append(best_iou)
        print(f"\nFold {fold + 1} Best IoU: {best_iou:.4f}")
    
    print(f"\n{'='*50}")
    print(f"K-Fold Cross Validation Results:")
    print(f"Mean IoU: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"Fold Results: {fold_results}")
    
    return fold_results

# Final ensemble prediction
def create_ensemble_prediction():
    """Load all fold models and create ensemble predictions"""
    models = []
    
    for fold in range(config.N_FOLDS):
        model = EnhancedSegmentationModel().to(device)
        checkpoint = torch.load(f'best_model_fold{fold}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    
    # Test dataset
    test_images = sorted([os.path.join(test_image_path, f) for f in os.listdir(test_image_path)])
    test_masks = sorted([os.path.join(test_mask_path, f) for f in os.listdir(test_mask_path)])
    
    test_dataset = SegmentationDataset(
        test_images, test_masks,
        transform=get_validation_augmentation()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    # TTA for each model
    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
        tta.Scale(scales=[0.9, 1.0, 1.1], interpolation='bilinear'),
    ])
    
    tta_models = [tta.SegmentationTTAWrapper(model, tta_transforms, merge_mode='mean') 
                  for model in models]
    
    # Ensemble prediction
    total_iou = 0
    total_iou_pp = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions from all models
            all_preds = []
            for tta_model in tta_models:
                outputs = tta_model(images)
                preds = torch.sigmoid(outputs)
                all_preds.append(preds)
            
            # Weighted ensemble (you can adjust weights based on fold performance)
            weights = torch.tensor([1.0] * len(all_preds)).to(device)
            weights = weights / weights.sum()
            
            ensemble_preds = sum(w * pred for w, pred in zip(weights, all_preds))
            
            # Calculate IoU
            iou = calculate_iou(ensemble_preds, masks)
            
            # With post-processing
            preds_pp = torch.stack([
                advanced_post_process(ensemble_preds[i, 0], config.MIN_OBJECT_SIZE, config.MORPHOLOGY_KERNEL)
                for i in range(ensemble_preds.shape[0])
            ]).unsqueeze(1).to(device)
            iou_pp = calculate_iou(preds_pp, masks)
            
            total_iou += iou.item()
            total_iou_pp += iou_pp.item()
    
    final_iou = total_iou / len(test_loader)
    final_iou_pp = total_iou_pp / len(test_loader)
    
    print(f"\n{'='*50}")
    print(f"FINAL ENSEMBLE RESULTS")
    print(f"IoU (without post-processing): {final_iou:.4f}")
    print(f"IoU (with post-processing): {final_iou_pp:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train with K-Fold
    fold_results = train_kfold()
    
    # Create ensemble predictions
    create_ensemble_prediction()