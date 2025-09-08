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

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ GPU is available!")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("⚠️ GPU not detected. Running on CPU.")

print(f"Using device: {device}")

# Paths
train_dir = '/kaggle/input/brisc2025/brisc2025/segmentation_task/train'
test_dir = '/kaggle/input/brisc2025/brisc2025/segmentation_task/test'

train_image_path = os.path.join(train_dir, 'images')
train_mask_path = os.path.join(train_dir, 'masks')
test_image_path = os.path.join(test_dir, 'images')
test_mask_path = os.path.join(test_dir, 'masks')

# Configuration
IMG_HEIGHT = 512  # Increased from 384
IMG_WIDTH = 512   # Increased from 384
BATCH_SIZE = 8    # Reduced for larger images
NUM_CLASSES = 1
EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Advanced Data Augmentation
def get_training_augmentation():
    return A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
        A.GridDistortion(p=0.3),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=64, max_width=64, min_holes=1, 
                        min_height=16, min_width=16, fill_value=0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# Enhanced Dataset with Augmentation
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0).float() / 255.0
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return image, mask

# Advanced Model Architecture
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        size = x.size()[2:]
        
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        x5 = self.avg_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        x5 = self.relu(x5)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.bn_out(self.conv_out(x)))
        return x

# Enhanced UNet++ with EfficientNet backbone
class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=1):
        super(UNetPlusPlus, self).__init__()
        # Using segmentation_models_pytorch for better performance
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
            decoder_attention_type="scse"  # Squeeze-and-Excitation + Channel Attention
        )
        
    def forward(self, x):
        return self.model(x)

# Advanced Loss Functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = targets.size(0)
        dims = (2, 3)
        
        intersection = (probs * targets).sum(dims)
        cardinality = (probs + targets).sum(dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return (1. - dice_score).mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        
        tp = (probs * targets).sum(dims)
        fp = (probs * (1 - targets)).sum(dims)
        fn = ((1 - probs) * targets).sum(dims)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        
        tp = (probs * targets).sum(dims)
        fp = (probs * (1 - targets)).sum(dims)
        fn = ((1 - probs) * targets).sum(dims)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky.mean()

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, ce_ratio=0.5, smooth=1.0):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.smooth = smooth

    def forward(self, logits, targets):
        targets = targets.float()
        
        # Dice loss
        dice = DiceLoss(self.smooth)(logits, targets)
        
        # BCE loss with positive weight
        pos_weight = (targets.numel() - targets.sum()) / (targets.sum() + 1e-7)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=torch.tensor(pos_weight).to(logits.device)
        )
        
        # Focal Tversky
        focal_tversky = FocalTverskyLoss()(logits, targets)
        
        # Combination
        return self.ce_ratio * bce + (1 - self.ce_ratio) * (self.alpha * dice + (1 - self.alpha) * focal_tversky)

# IoU calculation
def calculate_iou(pred, target, threshold=0.5, smooth=1e-6):
    pred = (pred > threshold).float()
    intersection = torch.sum(pred * target, dim=[1, 2, 3])
    union = torch.sum(pred + target, dim=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

# Post-processing functions
def remove_small_objects(mask, min_size=100):
    """Remove small connected components"""
    import cv2
    mask_np = mask.cpu().numpy().astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            labels[labels == i] = 0
    
    return torch.from_numpy((labels > 0).astype(np.float32)).to(mask.device)

def apply_morphology(mask, kernel_size=5):
    """Apply morphological operations to clean mask"""
    import cv2
    mask_np = mask.cpu().numpy().astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Close small holes
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Remove small objects
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return torch.from_numpy(mask_np.astype(np.float32)).to(mask.device)

def post_process_predictions(predictions, threshold=0.5, min_size=100):
    """Apply post-processing to predictions"""
    batch_size = predictions.shape[0]
    processed = []
    
    for i in range(batch_size):
        pred = (predictions[i, 0] > threshold).float()
        pred = apply_morphology(pred)
        pred = remove_small_objects(pred, min_size)
        processed.append(pred.unsqueeze(0))
    
    return torch.stack(processed)

# Training function with mixed precision
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    total_iou = 0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            iou = calculate_iou(preds, masks)
        
        total_loss += loss.item()
        total_iou += iou
    
    return total_loss / len(loader), total_iou / len(loader)

# Validation function
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            preds = torch.sigmoid(outputs)
            iou = calculate_iou(preds, masks)
            
            total_loss += loss.item()
            total_iou += iou
    
    return total_loss / len(loader), total_iou / len(loader)

# Main training script
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create datasets
    train_dataset = SegmentationDataset(
        train_image_path, train_mask_path, 
        transform=get_training_augmentation()
    )
    val_dataset = SegmentationDataset(
        test_image_path, test_mask_path,
        transform=get_validation_augmentation()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Initialize model
    model = UNetPlusPlus(num_classes=NUM_CLASSES).to(device)
    
    # Loss function
    criterion = ComboLoss(alpha=0.5, ce_ratio=0.3)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training history
    best_iou = 0
    patience = 30
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        # Training
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validation
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_model_improved.pth')
            print(f"✅ New best IoU: {best_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    
    print(f"Training complete! Best IoU: {best_iou:.4f}")
    
    # Test Time Augmentation for final predictions
    print("\nApplying Test Time Augmentation...")
    model.load_state_dict(torch.load('best_model_improved.pth'))
    
    tta_model = tta.SegmentationTTAWrapper(
        model, tta.aliases.d4_transform(), merge_mode='mean'
    )
    
    # Final validation with TTA
    tta_iou = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = tta_model(images)
            preds = torch.sigmoid(outputs)
            
            # Post-processing
            preds_processed = post_process_predictions(preds)
            
            iou = calculate_iou(preds_processed, masks)
            tta_iou += iou
    
    final_iou = tta_iou / len(val_loader)
    print(f"Final IoU with TTA and post-processing: {final_iou:.4f}")

if __name__ == "__main__":
    main()