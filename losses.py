"""
Advanced loss functions for medical image segmentation.
Combines multiple loss functions for better performance.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


# ---------------- Weighted BCE ----------------
class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss with automatic or fixed positive weight."""
    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        self.register_buffer(
            'static_pos_weight',
            torch.tensor(pos_weight) if pos_weight is not None else None
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if self.static_pos_weight is None:
            pos = targets.sum()
            neg = targets.numel() - pos
            pos_weight = neg / (pos + 1e-6)
            pos_weight_tensor = pos_weight.to(logits.device)
        else:
            pos_weight_tensor = self.static_pos_weight.to(logits.device)

        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight_tensor, reduction='mean'
        )


# ---------------- Dice Loss ----------------
class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()


# ---------------- Dice + BCE ----------------
class DiceBCELoss(nn.Module):
    """Combined Dice and BCE Loss."""
    def __init__(self, pos_weight: float | None = None,
                 bce_weight: float = 0.3, dice_weight: float = 0.7,
                 smooth: float = 1e-6) -> None:
        super().__init__()
        self.bce = WeightedBCELoss(pos_weight)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ---------------- Focal Tversky ----------------
class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for handling class imbalance."""
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 4 / 3, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1.0 - targets)).sum(dim=(2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        loss = torch.pow((1.0 - tversky), self.gamma)
        return loss.mean()


# ---------------- IoU Loss (Jaccard) ----------------
class IoULoss(nn.Module):
    """IoU (Jaccard) Loss for segmentation."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(2, 3))
        total = (probs + targets).sum(dim=(2, 3))
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()


# ---------------- Boundary Loss ----------------
class BoundaryLoss(nn.Module):
    """Boundary Loss using distance transform."""
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Compute distance maps for each sample in batch
        dist_maps = []
        for t in targets.cpu().numpy():
            t = t.squeeze().astype(np.uint8)
            # Distance transform for both foreground and background
            dt_fg = distance_transform_edt(1 - t)
            dt_bg = distance_transform_edt(t)
            dt = dt_fg + dt_bg
            dist_maps.append(dt)
        
        dist_maps = torch.tensor(np.stack(dist_maps), dtype=torch.float32, device=logits.device)
        multip = probs.squeeze(1) * dist_maps
        return multip.mean()


# ---------------- Focal Loss ----------------
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ---------------- Combined Loss (Dice+BCE + Tversky) ----------------
class CombinedLoss(nn.Module):
    """Combined loss using Dice+BCE and Focal Tversky."""
    def __init__(self, pos_weight: float | None = None,
                 dice_bce_weight: float = 0.5, focal_tversky_weight: float = 0.5,
                 dice_kwargs: dict | None = None,
                 tversky_kwargs: dict | None = None) -> None:
        super().__init__()
        dice_kwargs = dice_kwargs or {}
        tversky_kwargs = tversky_kwargs or {}
        self.dice_loss = DiceBCELoss(pos_weight=pos_weight, **dice_kwargs)
        self.tversky_loss = FocalTverskyLoss(**tversky_kwargs)
        total = dice_bce_weight + focal_tversky_weight
        self.w_dice = dice_bce_weight / total
        self.w_tv = focal_tversky_weight / total

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_dice = self.dice_loss(logits, targets)
        loss_tv = self.tversky_loss(logits, targets)
        return self.w_dice * loss_dice + self.w_tv * loss_tv


# ---------------- Ultimate Loss (All components) ----------------
class UltimateLoss(nn.Module):
    """Ultimate loss combining multiple loss functions for maximum performance."""
    def __init__(self, pos_weight=5.0,
                 w_dice=0.3, w_tversky=0.3, w_iou=0.2, w_boundary=0.2):
        super().__init__()
        self.dice_bce = DiceBCELoss(pos_weight=pos_weight)
        self.tversky = FocalTverskyLoss()
        self.iou = IoULoss()
        self.boundary = BoundaryLoss()
        
        # Normalize weights
        total = w_dice + w_tversky + w_iou + w_boundary
        self.w_dice = w_dice / total
        self.w_tversky = w_tversky / total
        self.w_iou = w_iou / total
        self.w_boundary = w_boundary / total

    def forward(self, logits, targets):
        loss1 = self.dice_bce(logits, targets)
        loss2 = self.tversky(logits, targets)
        loss3 = self.iou(logits, targets)
        loss4 = self.boundary(logits, targets)
        return (self.w_dice * loss1 +
                self.w_tversky * loss2 +
                self.w_iou * loss3 +
                self.w_boundary * loss4)


# ---------------- Adaptive Loss (Dynamic weighting) ----------------
class AdaptiveLoss(nn.Module):
    """Adaptive loss that adjusts weights based on training progress."""
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.dice_bce = DiceBCELoss(pos_weight=pos_weight)
        self.tversky = FocalTverskyLoss()
        self.iou = IoULoss()
        self.focal = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Learnable weights
        self.weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        
    def forward(self, logits, targets):
        # Softmax to ensure weights sum to 1
        w = F.softmax(self.weights, dim=0)
        
        loss1 = self.dice_bce(logits, targets)
        loss2 = self.tversky(logits, targets)
        loss3 = self.iou(logits, targets)
        loss4 = self.focal(logits, targets)
        
        return w[0] * loss1 + w[1] * loss2 + w[2] * loss3 + w[3] * loss4


__all__ = [
    "WeightedBCELoss", "DiceLoss", "DiceBCELoss", "FocalTverskyLoss", 
    "IoULoss", "BoundaryLoss", "FocalLoss", "CombinedLoss", "UltimateLoss", "AdaptiveLoss"
]