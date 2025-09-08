# بهبودهای سریع برای کد فعلی شما (بدون تغییر کامل معماری)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 1. آگمنتیشن قوی‌تر برای dataset فعلی
def get_strong_augmentation(image_size=(384, 384)):
    """آگمنتیشن‌های قوی برای dataset شما"""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        
        # Geometric
        A.OneOf([
            A.RandomRotate90(p=1),
            A.Flip(p=1),
            A.Transpose(p=1)
        ], p=0.7),
        
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
            A.GridDistortion(p=1),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.3),
        
        # Color
        A.OneOf([
            A.CLAHE(clip_limit=2, p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.HueSaturationValue(p=1)
        ], p=0.5),
        
        # Noise
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1),
        ], p=0.3),
        
        # Cutout
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        
        ToTensorV2()
    ])


# 2. بهبود Loss Function
class ImprovedCombinedLoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.smooth = 1e-6
        
    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()
    
    def tversky_loss(self, pred, target, alpha=0.3, beta=0.7):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + alpha * fn + beta * fp + self.smooth)
        return (1 - tversky).mean()
    
    def forward(self, pred, target):
        bce = self.bce(pred, target) * 0.2
        dice = self.dice_loss(pred, target) * 0.3
        focal = self.focal_loss(pred, target) * 0.25
        tversky = self.tversky_loss(pred, target) * 0.25
        
        return bce + dice + focal + tversky


# 3. بهبود مدل موجود با اضافه کردن Attention
class ImprovedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
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


# 4. Test Time Augmentation محسن شده
def enhanced_tta_predict(model, image, device):
    """TTA با 8 augmentation"""
    model.eval()
    predictions = []
    
    # Original
    with torch.no_grad():
        pred = torch.sigmoid(model(image))
        predictions.append(pred)
    
    # Rotations
    for k in [1, 2, 3]:  # 90, 180, 270 degrees
        rotated = torch.rot90(image, k, [2, 3])
        with torch.no_grad():
            pred = torch.sigmoid(model(rotated))
            pred = torch.rot90(pred, -k, [2, 3])
            predictions.append(pred)
    
    # Flips
    for dims in [[2], [3], [2, 3]]:  # vertical, horizontal, both
        flipped = torch.flip(image, dims)
        with torch.no_grad():
            pred = torch.sigmoid(model(flipped))
            pred = torch.flip(pred, dims)
            predictions.append(pred)
    
    # میانگین با وزن (وزن بیشتر به original)
    weights = [2.0] + [1.0] * 7  # Original gets double weight
    weights = torch.tensor(weights).to(device) / sum(weights)
    
    final_pred = sum(w * p for w, p in zip(weights, predictions))
    return final_pred


# 5. Training Loop بهینه شده
def optimized_training_step(model, images, masks, criterion, optimizer, scaler, augmenter):
    """یک step از training با تکنیک‌های بهینه"""
    
    # Online augmentation
    if np.random.rand() < 0.5:
        # MixUp
        alpha = 0.4
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(images.size(0)).to(images.device)
        
        images = lam * images + (1 - lam) * images[index]
        masks = lam * masks + (1 - lam) * masks[index]
    
    elif np.random.rand() < 0.5:
        # CutMix
        lam = np.random.beta(1.0, 1.0)
        index = torch.randperm(images.size(0)).to(images.device)
        
        H, W = images.shape[2:]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        masks[:, :, bby1:bby2, bbx1:bbx2] = masks[index, :, bby1:bby2, bbx1:bbx2]
    
    # Mixed precision forward
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, masks)
    
    # Backward with gradient scaling
    scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    return loss.item()


# 6. Learning Rate Scheduling بهتر
class WarmupPolyLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, power=0.9):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.power = power
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup
            factor = epoch / self.warmup_epochs
        else:
            # Polynomial decay
            factor = (1 - (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)) ** self.power
        
        for idx, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[idx] * factor


# 7. Post-processing
def post_process_predictions(pred_mask, min_size=100):
    """پردازش نهایی predictions"""
    import cv2
    from scipy.ndimage import binary_fill_holes
    
    # Convert to numpy
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    
    # Remove small objects
    num_labels, labels = cv2.connectedComponents((pred_mask > 0.5).astype(np.uint8))
    
    for label in range(1, num_labels):
        if np.sum(labels == label) < min_size:
            pred_mask[labels == label] = 0
    
    # Fill holes
    pred_mask = binary_fill_holes(pred_mask > 0.5).astype(np.float32)
    
    # Smooth boundaries with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
    
    return pred_mask


# 8. نمونه استفاده در training loop شما
"""
# جایگزین کردن loss
criterion = ImprovedCombinedLoss(pos_weight=5.0).to(device)

# اضافه کردن scheduler
warmup_scheduler = WarmupPolyLR(optimizer, warmup_epochs=5, total_epochs=EPOCHS)

# در training loop:
for epoch in range(EPOCHS):
    # Training
    for images, masks in train_loader:
        loss = optimized_training_step(model, images, masks, criterion, optimizer, scaler, None)
    
    # Update LR
    warmup_scheduler.step(epoch)
    
    # Validation with TTA
    model.eval()
    for images, masks in test_loader:
        predictions = enhanced_tta_predict(model, images, device)
        # Post-process if needed
        for i in range(predictions.shape[0]):
            predictions[i, 0] = post_process_predictions(predictions[i, 0])

"""

print("✅ Quick improvements ready to use!")
print("این بهبودها را می‌توانید مستقیماً در کد فعلی خود اعمال کنید.")