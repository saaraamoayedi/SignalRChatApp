import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import math

# Advanced Training Techniques

class WarmupCosineScheduler(_LRScheduler):
    """Cosine scheduler with linear warmup"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]

class SAM(torch.optim.Optimizer):
    """Sharpness Aware Minimization optimizer"""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
                
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

class MixUp:
    """MixUp augmentation for segmentation"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, masks):
        batch_size = images.size(0)
        indices = torch.randperm(batch_size).to(images.device)
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed_images = lam * images + (1 - lam) * images[indices]
        mixed_masks = lam * masks + (1 - lam) * masks[indices]
        
        return mixed_images, mixed_masks

class CutMix:
    """CutMix augmentation for segmentation"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, masks):
        batch_size = images.size(0)
        indices = torch.randperm(batch_size).to(images.device)
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        H, W = images.size(-2), images.size(-1)
        cut_h = int(H * np.sqrt(1 - lam))
        cut_w = int(W * np.sqrt(1 - lam))
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_images = images.clone()
        mixed_masks = masks.clone()
        
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        mixed_masks[:, :, bby1:bby2, bbx1:bbx2] = masks[indices, :, bby1:bby2, bbx1:bbx2]
        
        return mixed_images, mixed_masks

class PuzzleMix:
    """PuzzleMix: more sophisticated mixing strategy"""
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, images, masks):
        batch_size = images.size(0)
        indices = torch.randperm(batch_size).to(images.device)
        
        # Generate mixing masks based on saliency
        with torch.no_grad():
            # Simple saliency: use mask areas
            saliency = masks.squeeze(1)
            
            # Create puzzle pieces
            H, W = images.size(-2), images.size(-1)
            n_patches = 4  # 2x2 grid
            patch_h, patch_w = H // 2, W // 2
            
            mixed_images = images.clone()
            mixed_masks = masks.clone()
            
            for i in range(2):
                for j in range(2):
                    if np.random.random() > 0.5:
                        y1, y2 = i * patch_h, (i + 1) * patch_h
                        x1, x2 = j * patch_w, (j + 1) * patch_w
                        
                        mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
                        mixed_masks[:, :, y1:y2, x1:x2] = masks[indices, :, y1:y2, x1:x2]
        
        return mixed_images, mixed_masks

# Self-training / Pseudo-labeling
class PseudoLabeler:
    def __init__(self, model, threshold=0.9, min_area=100):
        self.model = model
        self.threshold = threshold
        self.min_area = min_area
    
    @torch.no_grad()
    def generate_pseudo_labels(self, unlabeled_images):
        self.model.eval()
        outputs = self.model(unlabeled_images)
        probs = torch.sigmoid(outputs)
        
        # High confidence predictions
        confident_mask = (probs > self.threshold) | (probs < (1 - self.threshold))
        pseudo_masks = (probs > 0.5).float()
        
        # Only keep high confidence regions
        pseudo_masks[~confident_mask] = 0.5  # Uncertain regions
        
        return pseudo_masks, confident_mask

# Knowledge Distillation
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, targets, base_criterion):
        # Distillation loss
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = self.criterion(student_probs, teacher_probs) * (self.temperature ** 2)
        
        # Student loss
        student_loss = base_criterion(student_logits, targets)
        
        # Combined loss
        return self.alpha * distillation_loss + (1 - self.alpha) * student_loss

# Stochastic Weight Averaging (SWA)
class SWAModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.swa_model = torch.optim.swa_utils.AveragedModel(model)
        self.swa_start = 100  # Start SWA after this epoch
        self.swa_freq = 5     # Update SWA model every N epochs
    
    def update_swa(self, epoch):
        if epoch >= self.swa_start and epoch % self.swa_freq == 0:
            self.swa_model.update_parameters(self.model)
    
    def get_swa_model(self):
        return self.swa_model

# Gradient Accumulation
class GradientAccumulator:
    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.steps = 0
    
    def should_step(self):
        self.steps += 1
        if self.steps % self.accumulation_steps == 0:
            return True
        return False
    
    def scale_loss(self, loss):
        return loss / self.accumulation_steps

# Advanced ensemble predictions
class EnsemblePredictor:
    def __init__(self, models):
        self.models = models
    
    @torch.no_grad()
    def predict(self, images, method='mean'):
        predictions = []
        
        for model in self.models:
            model.eval()
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predictions.append(probs)
        
        predictions = torch.stack(predictions)
        
        if method == 'mean':
            return predictions.mean(dim=0)
        elif method == 'median':
            return predictions.median(dim=0)[0]
        elif method == 'weighted':
            # Weight by model performance
            weights = torch.tensor([0.3, 0.3, 0.4]).to(predictions.device)
            weights = weights.view(-1, 1, 1, 1, 1)
            return (predictions * weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

# Multi-scale training and inference
class MultiScaleWrapper(nn.Module):
    def __init__(self, model, scales=[0.75, 1.0, 1.25]):
        super().__init__()
        self.model = model
        self.scales = scales
    
    def forward(self, x, training=True):
        if training and np.random.random() > 0.5:
            # Random scale during training
            scale = np.random.choice(self.scales)
            if scale != 1.0:
                size = (int(x.shape[-2] * scale), int(x.shape[-1] * scale))
                x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
                output = self.model(x)
                output = F.interpolate(output, size=(x.shape[-2], x.shape[-1]), 
                                     mode='bilinear', align_corners=False)
                return output
        elif not training:
            # Multi-scale inference
            outputs = []
            for scale in self.scales:
                if scale != 1.0:
                    size = (int(x.shape[-2] * scale), int(x.shape[-1] * scale))
                    scaled_x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
                    output = self.model(scaled_x)
                    output = F.interpolate(output, size=(x.shape[-2], x.shape[-1]), 
                                         mode='bilinear', align_corners=False)
                else:
                    output = self.model(x)
                outputs.append(output)
            return torch.mean(torch.stack(outputs), dim=0)
        
        return self.model(x)

# Label Smoothing for segmentation
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)

# Deep Supervision
class DeepSupervisionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.side_outputs = nn.ModuleList([
            nn.Conv2d(256, 1, 1),  # For intermediate feature maps
            nn.Conv2d(128, 1, 1),
            nn.Conv2d(64, 1, 1),
        ])
    
    def forward(self, x):
        # This would need modification based on your model architecture
        # to extract intermediate features
        main_output = self.model(x)
        return main_output

# Contrastive Learning for Segmentation
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, masks):
        # Pixel-level contrastive learning
        batch_size, channels, height, width = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, channels)
        masks = masks.reshape(-1)
        
        # Sample positive and negative pairs
        pos_mask = masks == 1
        neg_mask = masks == 0
        
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_features = features[pos_mask]
            neg_features = features[neg_mask]
            
            # Compute similarities
            pos_sim = torch.matmul(pos_features, pos_features.T) / self.temperature
            neg_sim = torch.matmul(pos_features, neg_features.T) / self.temperature
            
            # Contrastive loss
            labels = torch.zeros(pos_sim.shape[0]).long().to(features.device)
            logits = torch.cat([pos_sim, neg_sim], dim=1)
            
            loss = F.cross_entropy(logits, labels)
            return loss
        
        return torch.tensor(0.0).to(features.device)