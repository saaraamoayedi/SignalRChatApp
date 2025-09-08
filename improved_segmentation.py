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
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import random
import time

# تنظیمات CUDA برای بهینه‌سازی
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ GPU is available!")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("⚠️ GPU not detected. Running on CPU.")

# تنظیمات عمومی بهینه شده
IMG_HEIGHT = 512  # افزایش سایز برای جزئیات بیشتر
IMG_WIDTH = 512
BATCH_SIZE = 8   # کاهش به خاطر سایز بزرگتر
NUM_CLASSES = 1
EPOCHS = 250
INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-4

# مسیرها
train_image_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/train/images'
train_mask_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/train/masks'
test_image_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/test/images'
test_mask_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/test/masks'


class HeavyAugmentation:
    """آگمنتیشن‌های قوی و متنوع"""
    def __init__(self, img_size=(512, 512), is_train=True):
        if is_train:
            self.transform = A.Compose([
                # تغییر سایز با حفظ نسبت
                A.LongestMaxSize(max_size=max(img_size), p=1.0),
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], 
                             border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                
                # آگمنتیشن‌های هندسی
                A.OneOf([
                    A.RandomRotate90(p=1.0),
                    A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                    A.Flip(p=1.0),
                ], p=0.7),
                
                # Deformation قوی
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
                ], p=0.5),
                
                # آگمنتیشن‌های رنگی (فقط روی تصویر)
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                ], p=0.8),
                
                # Noise & Blur
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                ], p=0.4),
                
                # Advanced augmentations
                A.OneOf([
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
                    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
                                  num_shadows_upper=2, shadow_dimension=5, p=1.0),
                ], p=0.3),
                
                # Cutout/CoarseDropout
                A.CoarseDropout(max_holes=8, max_height=64, max_width=64, 
                               min_holes=1, min_height=16, min_width=16,
                               fill_value=0, mask_fill_value=0, p=0.5),
                
                # Normalization
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            # Validation/Test augmentation
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=max(img_size), p=1.0),
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], 
                             border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    
    def __call__(self, image, mask):
        # اطمینان از uint8 بودن
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
            
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask'].float() / 255.0


class AdvancedSegmentationDataset(Dataset):
    """دیتاست با پشتیبانی از TTA و آگمنتیشن پیشرفته"""
    def __init__(self, image_dir, mask_dir, transform=None, tta_transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.tta_transforms = tta_transforms
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
            image, mask = self.transform(image, mask)
        
        # اضافه کردن بُعد channel به mask
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask


class AttentionBlock(nn.Module):
    """Self-Attention Block برای بهبود feature extraction"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Query, Key, Value
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        # Attention
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling برای multi-scale features"""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        # Different dilation rates
        dilations = [1, 6, 12, 18]
        
        self.aspp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1 if dilation == 1 else 3,
                         padding=0 if dilation == 1 else dilation, dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for dilation in dilations
        ])
        
        # Global average pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        size = x.shape[2:]
        
        # Apply ASPP blocks
        feat_list = []
        for aspp_block in self.aspp_blocks:
            feat_list.append(aspp_block(x))
            
        # Global pooling
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        feat_list.append(global_feat)
        
        # Concatenate all features
        x = torch.cat(feat_list, dim=1)
        x = self.conv1(x)
        
        return x


class HybridModel(nn.Module):
    """ترکیب چند معماری قوی با Ensemble"""
    def __init__(self, num_classes=1):
        super().__init__()
        
        # مدل اصلی: DeepLabV3+ با EfficientNet-B5
        self.deeplabv3plus = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b5",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
            decoder_atrous_rates=(6, 12, 18),
            decoder_channels=512
        )
        
        # مدل کمکی: UNet++ با SE-ResNeXt101
        self.unetplusplus = smp.UnetPlusPlus(
            encoder_name="se_resnext101_32x4d",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
            decoder_attention_type="scse"
        )
        
        # ASPP module برای features بهتر
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        
        # Self-Attention
        self.attention = AttentionBlock(512)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(num_classes * 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
        
    def forward(self, x):
        # DeepLabV3+ prediction
        pred1 = self.deeplabv3plus(x)
        
        # UNet++ prediction
        pred2 = self.unetplusplus(x)
        
        # Fusion with learned weights
        fused = torch.cat([pred1, pred2], dim=1)
        output = self.fusion(fused)
        
        return output


class CombinedSegmentationLoss(nn.Module):
    """ترکیب بهینه loss functions مختلف"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal_loss = smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0)
        self.tversky_loss = smp.losses.TverskyLoss(mode='binary', from_logits=True, 
                                                   alpha=0.3, beta=0.7, gamma=1.5)
        self.lovasz_loss = smp.losses.LovaszLoss(mode='binary', from_logits=True)
        
    def forward(self, pred, target):
        # وزن‌های بهینه شده
        bce = self.bce(pred, target) * 0.2
        dice = self.dice_loss(pred, target) * 0.25
        focal = self.focal_loss(pred, target) * 0.2
        tversky = self.tversky_loss(pred, target) * 0.2
        lovasz = self.lovasz_loss(pred, target) * 0.15
        
        return bce + dice + focal + tversky + lovasz


class PolyLRScheduler:
    """Polynomial Learning Rate Decay"""
    def __init__(self, optimizer, max_epochs, power=0.9):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.power = power
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        decay = (1 - epoch / self.max_epochs) ** self.power
        for idx, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[idx] * decay
            
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    batch_size, _, h, w = x.size()
    indices = torch.randperm(batch_size).to(x.device)
    
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)
    
    # Random position
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    x[:, :, bby1:bby2, bbx1:bbx2] = x[indices, :, bby1:bby2, bbx1:bbx2]
    y[:, :, bby1:bby2, bbx1:bbx2] = y[indices, :, bby1:bby2, bbx1:bbx2]
    
    return x, y


def calculate_metrics(pred, target, threshold=0.5):
    """محاسبه معیارهای مختلف"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()
    
    # IoU
    intersection = (pred_binary * target_binary).sum(dim=[1, 2, 3])
    union = (pred_binary + target_binary).sum(dim=[1, 2, 3]) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Dice
    dice = (2 * intersection + 1e-6) / (pred_binary.sum(dim=[1, 2, 3]) + target_binary.sum(dim=[1, 2, 3]) + 1e-6)
    
    # Pixel Accuracy
    correct = (pred_binary == target_binary).float().sum(dim=[1, 2, 3])
    total = target_binary.numel() / target_binary.size(0)
    pixel_acc = correct / total
    
    return {
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'pixel_acc': pixel_acc.mean().item()
    }


# تنظیم seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# آماده‌سازی داده‌ها
train_transform = HeavyAugmentation(img_size=(IMG_HEIGHT, IMG_WIDTH), is_train=True)
val_transform = HeavyAugmentation(img_size=(IMG_HEIGHT, IMG_WIDTH), is_train=False)

train_dataset = AdvancedSegmentationDataset(
    train_image_path, train_mask_path, transform=train_transform
)
test_dataset = AdvancedSegmentationDataset(
    test_image_path, test_mask_path, transform=val_transform
)

# DataLoaders با workers بیشتر
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)

# مدل و بهینه‌ساز
model = HybridModel(num_classes=NUM_CLASSES).to(device)
criterion = CombinedSegmentationLoss().to(device)

# Optimizer با تنظیمات مختلف برای encoder و decoder
encoder_params = []
decoder_params = []

for name, param in model.named_parameters():
    if 'encoder' in name:
        encoder_params.append(param)
    else:
        decoder_params.append(param)

optimizer = optim.AdamW([
    {'params': encoder_params, 'lr': INITIAL_LR * 0.1},  # Lower LR for pretrained encoder
    {'params': decoder_params, 'lr': INITIAL_LR}
], weight_decay=WEIGHT_DECAY)

# Scheduler
scheduler = PolyLRScheduler(optimizer, max_epochs=EPOCHS, power=0.9)

# Mixed Precision Training
scaler = GradScaler()

# Training parameters
best_iou = 0
patience = 30
patience_counter = 0
train_losses, train_ious = [], []
val_losses, val_ious = [], []

print("🚀 شروع آموزش با تنظیمات پیشرفته...")
print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Training loop
for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    train_metrics = {'iou': 0, 'dice': 0, 'pixel_acc': 0}
    
    pbar = enumerate(train_loader)
    for batch_idx, (images, masks) in pbar:
        images, masks = images.to(device), masks.to(device)
        
        # Random augmentation selection
        aug_choice = random.random()
        if aug_choice < 0.25:
            images, masks = mixup_data(images, masks, alpha=0.4)
        elif aug_choice < 0.5:
            images, masks = cutmix_data(images, masks, alpha=1.0)
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # Gradient accumulation (simulate larger batch)
        loss = loss / 2  # accumulate over 2 steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % 2 == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics
        with torch.no_grad():
            metrics = calculate_metrics(outputs, masks)
            for key in train_metrics:
                train_metrics[key] += metrics[key]
        
        train_loss += loss.item() * 2  # compensate for division
        
        # Progress
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item()*2:.4f} IoU: {metrics['iou']:.4f}")
    
    # Average metrics
    train_loss /= len(train_loader)
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)
    
    # Validation with TTA
    model.eval()
    val_loss = 0
    val_metrics = {'iou': 0, 'dice': 0, 'pixel_acc': 0}
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Test Time Augmentation
            outputs_list = []
            
            # Original
            with autocast():
                outputs_list.append(torch.sigmoid(model(images)))
            
            # Horizontal flip
            with autocast():
                outputs_list.append(torch.sigmoid(model(torch.flip(images, dims=[3]))))
            outputs_list[-1] = torch.flip(outputs_list[-1], dims=[3])
            
            # Vertical flip
            with autocast():
                outputs_list.append(torch.sigmoid(model(torch.flip(images, dims=[2]))))
            outputs_list[-1] = torch.flip(outputs_list[-1], dims=[2])
            
            # Average predictions
            outputs = torch.stack(outputs_list).mean(dim=0)
            outputs_logits = torch.logit(outputs.clamp(1e-7, 1-1e-7))
            
            loss = criterion(outputs_logits, masks)
            val_loss += loss.item()
            
            metrics = calculate_metrics(outputs_logits, masks)
            for key in val_metrics:
                val_metrics[key] += metrics[key]
    
    # Average validation metrics
    val_loss /= len(test_loader)
    for key in val_metrics:
        val_metrics[key] /= len(test_loader)
    
    # Update scheduler
    scheduler.step(epoch)
    current_lr = scheduler.get_last_lr()[0]
    
    # Logging
    train_losses.append(train_loss)
    train_ious.append(train_metrics['iou'])
    val_losses.append(val_loss)
    val_ious.append(val_metrics['iou'])
    
    print(f"\n📊 Epoch [{epoch+1}/{EPOCHS}] Summary:")
    print(f"Train - Loss: {train_loss:.4f}, IoU: {train_metrics['iou']:.4f}, "
          f"Dice: {train_metrics['dice']:.4f}, Acc: {train_metrics['pixel_acc']:.4f}")
    print(f"Val - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, "
          f"Dice: {val_metrics['dice']:.4f}, Acc: {val_metrics['pixel_acc']:.4f}")
    print(f"LR: {current_lr:.6f}")
    
    # Save best model
    if val_metrics['iou'] > best_iou:
        best_iou = val_metrics['iou']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
            'scaler': scaler.state_dict()
        }, 'best_model_advanced.pth')
        print(f"🎉 New best IoU: {best_iou:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        
    # Early stopping
    if patience_counter >= patience:
        print(f"⏹ Early stopping at epoch {epoch+1}")
        break
    
    # Plot progress
    if (epoch + 1) % 5 == 0:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_ious, label='Train IoU')
        plt.plot(val_ious, label='Val IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_progress_epoch_{epoch+1}.png')
        plt.close()

print(f"\n✅ Training completed! Best IoU: {best_iou:.4f}")