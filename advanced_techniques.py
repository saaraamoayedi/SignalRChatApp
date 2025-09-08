import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PseudoLabelingStrategy:
    """استراتژی Pseudo-Labeling برای استفاده از داده‌های بدون لیبل"""
    def __init__(self, model, confidence_threshold=0.95):
        self.model = model
        self.confidence_threshold = confidence_threshold
        
    def generate_pseudo_labels(self, unlabeled_loader, device):
        """تولید pseudo-labels با confidence بالا"""
        self.model.eval()
        pseudo_data = []
        
        with torch.no_grad():
            for images, _ in unlabeled_loader:
                images = images.to(device)
                
                # TTA برای پیش‌بینی قوی‌تر
                predictions = []
                
                # Original
                pred = torch.sigmoid(self.model(images))
                predictions.append(pred)
                
                # Flips
                for flip_dims in [[2], [3], [2, 3]]:
                    flipped = torch.flip(images, dims=flip_dims)
                    pred = torch.sigmoid(self.model(flipped))
                    pred = torch.flip(pred, dims=flip_dims)
                    predictions.append(pred)
                
                # میانگین پیش‌بینی‌ها
                final_pred = torch.stack(predictions).mean(dim=0)
                
                # فقط نواحی با confidence بالا
                confident_mask = (final_pred > self.confidence_threshold) | (final_pred < (1 - self.confidence_threshold))
                
                for i in range(images.size(0)):
                    if confident_mask[i].sum() > 0.7 * confident_mask[i].numel():
                        pseudo_data.append({
                            'image': images[i].cpu(),
                            'mask': (final_pred[i] > 0.5).float().cpu(),
                            'confidence': confident_mask[i].float().mean().item()
                        })
        
        return pseudo_data


class SelfEnsembleModel:
    """Self-Ensemble با Stochastic Weight Averaging (SWA)"""
    def __init__(self, base_model, swa_start=150, swa_lr=1e-4):
        self.base_model = base_model
        self.swa_model = torch.optim.swa_utils.AveragedModel(base_model)
        self.swa_start = swa_start
        self.swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer=None,  # تنظیم می‌شود
            swa_lr=swa_lr
        )
        
    def update_swa(self, epoch):
        """بروزرسانی SWA model"""
        if epoch >= self.swa_start:
            self.swa_model.update_parameters(self.base_model)
            
    def get_ensemble_prediction(self, x, num_forward=5):
        """چند forward pass با dropout فعال"""
        self.base_model.train()  # فعال کردن dropout
        predictions = []
        
        for _ in range(num_forward):
            with torch.no_grad():
                pred = torch.sigmoid(self.base_model(x))
                predictions.append(pred)
        
        # میانگین و عدم قطعیت
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty


class MultiScaleInference:
    """استنتاج در مقیاس‌های مختلف"""
    def __init__(self, model, scales=[0.75, 1.0, 1.25, 1.5]):
        self.model = model
        self.scales = scales
        
    def predict(self, image, original_size):
        """پیش‌بینی multi-scale"""
        self.model.eval()
        predictions = []
        
        for scale in self.scales:
            # تغییر سایز
            scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            
            if scale != 1.0:
                scaled_img = F.interpolate(image, size=scaled_size, mode='bilinear', align_corners=False)
            else:
                scaled_img = image
            
            # پیش‌بینی
            with torch.no_grad():
                pred = torch.sigmoid(self.model(scaled_img))
            
            # برگرداندن به سایز اصلی
            if scale != 1.0:
                pred = F.interpolate(pred, size=original_size, mode='bilinear', align_corners=False)
            
            predictions.append(pred)
        
        # میانگین وزن‌دار (وزن بیشتر به مقیاس‌های نزدیک 1.0)
        weights = torch.tensor([np.exp(-abs(s - 1.0)) for s in self.scales]).to(image.device)
        weights = weights / weights.sum()
        
        final_pred = sum(w * p for w, p in zip(weights, predictions))
        return final_pred


class OnlineBatchAugmentation:
    """آگمنتیشن‌های سنگین در حین training"""
    def __init__(self, p=0.5):
        self.p = p
        self.augmentations = [
            self.mosaic_augmentation,
            self.gridmask_augmentation,
            self.copy_paste_augmentation,
            self.style_transfer_augmentation
        ]
    
    def mosaic_augmentation(self, images, masks, batch_size=4):
        """ترکیب 4 تصویر در یک تصویر mosaic"""
        if batch_size < 4:
            return images, masks
            
        # انتخاب 4 تصویر تصادفی
        indices = torch.randperm(images.size(0))[:4]
        
        # ساخت mosaic
        h, w = images.shape[2:]
        mosaic_img = torch.zeros_like(images[0])
        mosaic_mask = torch.zeros_like(masks[0])
        
        # قرار دادن در 4 ربع
        mosaic_img[:, :h//2, :w//2] = F.interpolate(images[indices[0]].unsqueeze(0), size=(h//2, w//2))[0]
        mosaic_img[:, :h//2, w//2:] = F.interpolate(images[indices[1]].unsqueeze(0), size=(h//2, w//2))[0]
        mosaic_img[:, h//2:, :w//2] = F.interpolate(images[indices[2]].unsqueeze(0), size=(h//2, w//2))[0]
        mosaic_img[:, h//2:, w//2:] = F.interpolate(images[indices[3]].unsqueeze(0), size=(h//2, w//2))[0]
        
        mosaic_mask[:, :h//2, :w//2] = F.interpolate(masks[indices[0]].unsqueeze(0), size=(h//2, w//2))[0]
        mosaic_mask[:, :h//2, w//2:] = F.interpolate(masks[indices[1]].unsqueeze(0), size=(h//2, w//2))[0]
        mosaic_mask[:, h//2:, :w//2] = F.interpolate(masks[indices[2]].unsqueeze(0), size=(h//2, w//2))[0]
        mosaic_mask[:, h//2:, w//2:] = F.interpolate(masks[indices[3]].unsqueeze(0), size=(h//2, w//2))[0]
        
        # جایگزینی یک نمونه تصادفی
        idx = torch.randint(0, images.size(0), (1,)).item()
        images[idx] = mosaic_img
        masks[idx] = mosaic_mask
        
        return images, masks
    
    def gridmask_augmentation(self, images, masks, d_range=(96, 224), ratio=0.5):
        """GridMask augmentation"""
        b, c, h, w = images.shape
        
        for i in range(b):
            if torch.rand(1) < 0.5:
                # پارامترهای تصادفی
                d = torch.randint(d_range[0], d_range[1], (1,)).item()
                delta = torch.randint(0, d, (1,)).item()
                
                # ساخت grid mask
                grid_mask = torch.ones(h, w)
                for j in range(delta, h, d):
                    for k in range(delta, w, d):
                        if j + int(d * ratio) <= h and k + int(d * ratio) <= w:
                            grid_mask[j:j+int(d*ratio), k:k+int(d*ratio)] = 0
                
                # اعمال به تصویر و mask
                images[i] *= grid_mask.to(images.device)
                masks[i] *= grid_mask.to(masks.device)
        
        return images, masks
    
    def copy_paste_augmentation(self, images, masks):
        """Copy-Paste augmentation: کپی اشیاء از یک تصویر به دیگری"""
        b = images.size(0)
        if b < 2:
            return images, masks
        
        for i in range(0, b-1, 2):
            if torch.rand(1) < 0.5:
                # پیدا کردن نواحی شیء در mask
                obj_mask = masks[i] > 0.5
                
                if obj_mask.sum() > 100:  # حداقل اندازه شیء
                    # کپی شیء به تصویر بعدی
                    images[i+1] = torch.where(obj_mask, images[i], images[i+1])
                    masks[i+1] = torch.max(masks[i+1], masks[i] * obj_mask.float())
        
        return images, masks
    
    def style_transfer_augmentation(self, images, masks):
        """انتقال استایل ساده بین تصاویر batch"""
        b = images.size(0)
        if b < 2:
            return images, masks
        
        for i in range(b):
            if torch.rand(1) < 0.3:
                # انتخاب تصویر تصادفی برای style
                j = torch.randint(0, b, (1,)).item()
                if i != j:
                    # انتقال رنگ و کنتراست
                    mean_i = images[i].mean(dim=[1, 2], keepdim=True)
                    std_i = images[i].std(dim=[1, 2], keepdim=True)
                    mean_j = images[j].mean(dim=[1, 2], keepdim=True)
                    std_j = images[j].std(dim=[1, 2], keepdim=True)
                    
                    # Normalize و denormalize با آمار تصویر دیگر
                    images[i] = (images[i] - mean_i) / (std_i + 1e-6) * std_j + mean_j
        
        return images, masks
    
    def __call__(self, images, masks):
        """اعمال آگمنتیشن‌های تصادفی"""
        if torch.rand(1) < self.p:
            aug_func = np.random.choice(self.augmentations)
            images, masks = aug_func(images, masks)
        return images, masks


class KnowledgeDistillation:
    """Knowledge Distillation از مدل‌های قوی‌تر"""
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher.eval()  # Teacher همیشه در حالت eval
        
    def distillation_loss(self, student_logits, teacher_logits, targets, criterion):
        """محاسبه loss با KD"""
        # Soft targets از teacher
        teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
        student_probs = torch.sigmoid(student_logits / self.temperature)
        
        # KL divergence loss
        kd_loss = F.kl_div(
            student_probs.log(), 
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Student loss
        student_loss = criterion(student_logits, targets)
        
        # ترکیب losses
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * student_loss
        
        return total_loss


class ContrastiveLearning:
    """Contrastive learning برای بهبود representation"""
    def __init__(self, projection_dim=128, temperature=0.5):
        self.projection_dim = projection_dim
        self.temperature = temperature
        
    def contrastive_loss(self, features1, features2):
        """SimCLR-style contrastive loss"""
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        batch_size = features1.size(0)
        
        # ترکیب features
        features = torch.cat([features1, features2], dim=0)
        
        # محاسبه similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # ماسک برای جفت‌های مثبت
        mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)
        mask = mask.repeat(2, 2)
        mask[:batch_size, batch_size:].fill_diagonal_(True)
        mask[batch_size:, :batch_size].fill_diagonal_(True)
        
        # حذف خود-تشابهات
        similarity.masked_fill_(torch.eye(2 * batch_size, dtype=torch.bool).to(features.device), -float('inf'))
        
        # محاسبه loss
        pos_sim = similarity[mask].view(2 * batch_size, -1)
        neg_sim = similarity[~mask].view(2 * batch_size, -1)
        
        loss = -torch.log(
            torch.exp(pos_sim).sum(dim=1) / 
            torch.exp(torch.cat([pos_sim, neg_sim], dim=1)).sum(dim=1)
        ).mean()
        
        return loss


# تابع ترکیبی برای استفاده از همه تکنیک‌ها
def create_advanced_training_setup(base_model, train_loader, val_loader, device):
    """راه‌اندازی کامل تکنیک‌های پیشرفته"""
    
    # مدل‌های مختلف
    ensemble = SelfEnsembleModel(base_model)
    multi_scale = MultiScaleInference(base_model)
    augmenter = OnlineBatchAugmentation(p=0.5)
    
    # در صورت داشتن مدل teacher
    # teacher_model = load_pretrained_teacher()
    # kd = KnowledgeDistillation(teacher_model, base_model)
    
    # Pseudo labeling (در صورت داشتن داده بدون لیبل)
    # pseudo_labeler = PseudoLabelingStrategy(base_model)
    
    return {
        'ensemble': ensemble,
        'multi_scale': multi_scale,
        'augmenter': augmenter,
        # 'knowledge_distillation': kd,
        # 'pseudo_labeler': pseudo_labeler
    }


print("✅ تکنیک‌های پیشرفته آماده استفاده هستند!")