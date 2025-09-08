"""
Advanced techniques and improvements to boost IoU beyond 84%.
These are cutting-edge methods for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
import cv2
from scipy import ndimage


class TestTimeAugmentation:
    """Test-Time Augmentation for better inference results."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def predict_with_tta(self, image, num_augments=8):
        """Apply TTA and average predictions."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # Original
            pred = torch.sigmoid(self.model(image))
            predictions.append(pred)
            
            # Horizontal flip
            pred = torch.sigmoid(self.model(torch.flip(image, [3])))
            predictions.append(torch.flip(pred, [3]))
            
            # Vertical flip
            pred = torch.sigmoid(self.model(torch.flip(image, [2])))
            predictions.append(torch.flip(pred, [2]))
            
            # Both flips
            pred = torch.sigmoid(self.model(torch.flip(image, [2, 3])))
            predictions.append(torch.flip(pred, [2, 3]))
            
            # Rotations (90, 180, 270 degrees)
            for k in [1, 2, 3]:
                rotated = torch.rot90(image, k=k, dims=[2, 3])
                pred = torch.sigmoid(self.model(rotated))
                pred = torch.rot90(pred, k=-k, dims=[2, 3])
                predictions.append(pred)
            
            # Scale variations if requested
            if num_augments > 7:
                scales = [0.9, 1.1]
                for scale in scales[:num_augments-7]:
                    h, w = image.shape[2], image.shape[3]
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled = F.interpolate(image, size=(new_h, new_w), mode='bilinear')
                    pred = torch.sigmoid(self.model(scaled))
                    pred = F.interpolate(pred, size=(h, w), mode='bilinear')
                    predictions.append(pred)
        
        # Average all predictions
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred


class PostProcessing:
    """Advanced post-processing techniques."""
    
    @staticmethod
    def remove_small_objects(mask, min_size=100):
        """Remove small connected components."""
        mask_np = mask.cpu().numpy().squeeze()
        labeled = ndimage.label(mask_np > 0.5)[0]
        
        # Remove small objects
        for label_id in range(1, labeled.max() + 1):
            if np.sum(labeled == label_id) < min_size:
                labeled[labeled == label_id] = 0
        
        return torch.from_numpy((labeled > 0).astype(np.float32)).unsqueeze(0).to(mask.device)
    
    @staticmethod
    def fill_holes(mask):
        """Fill holes in segmentation mask."""
        mask_np = mask.cpu().numpy().squeeze()
        filled = ndimage.binary_fill_holes(mask_np > 0.5)
        return torch.from_numpy(filled.astype(np.float32)).unsqueeze(0).to(mask.device)
    
    @staticmethod
    def morphological_operations(mask, operation='close', kernel_size=3):
        """Apply morphological operations."""
        mask_np = (mask.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'close':
            result = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            result = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
        elif operation == 'dilate':
            result = cv2.dilate(mask_np, kernel, iterations=1)
        elif operation == 'erode':
            result = cv2.erode(mask_np, kernel, iterations=1)
        
        return torch.from_numpy(result.astype(np.float32)).unsqueeze(0).to(mask.device)
    
    @staticmethod
    def conditional_random_field(image, mask, num_iterations=5):
        """Simple CRF-like post-processing using bilateral filtering."""
        # Convert to numpy
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        mask_np = mask.cpu().numpy().squeeze()
        
        # Apply bilateral filter
        for _ in range(num_iterations):
            mask_np = cv2.bilateralFilter(mask_np.astype(np.float32), 9, 75, 75)
        
        return torch.from_numpy(mask_np).unsqueeze(0).to(mask.device)


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for better performance."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models)
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(torch.sigmoid(pred))
        
        # Weighted average
        ensemble_pred = sum(w * pred for w, pred in zip(self.weights, predictions))
        return torch.logit(ensemble_pred.clamp(1e-7, 1 - 1e-7))


class AdversarialTraining:
    """Adversarial training for robustness."""
    
    def __init__(self, model, epsilon=0.01, alpha=0.002, num_steps=7):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def generate_adversarial_examples(self, images, labels, criterion):
        """Generate adversarial examples using PGD."""
        images = images.clone().detach().requires_grad_(True)
        
        for _ in range(self.num_steps):
            images.requires_grad_(True)
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
            
            # Update adversarial examples
            adv_images = images + self.alpha * grad.sign()
            eta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            images = torch.clamp(images + eta, min=0, max=1).detach_()
        
        return images


class FocalLoss2D(nn.Module):
    """Improved 2D Focal Loss with better numerical stability."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t
        p_t = torch.exp(-bce_loss)
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SelfTraining:
    """Self-training with pseudo-labels for semi-supervised learning."""
    
    def __init__(self, model, confidence_threshold=0.9):
        self.model = model
        self.confidence_threshold = confidence_threshold
    
    def generate_pseudo_labels(self, unlabeled_images):
        """Generate pseudo-labels for unlabeled data."""
        self.model.eval()
        pseudo_labels = []
        confident_images = []
        
        with torch.no_grad():
            for images in unlabeled_images:
                outputs = torch.sigmoid(self.model(images))
                
                # Select confident predictions
                confidence = torch.max(torch.stack([outputs, 1 - outputs]), dim=0)[0]
                confident_mask = confidence > self.confidence_threshold
                
                if confident_mask.sum() > 0:
                    pseudo_label = (outputs > 0.5).float()
                    pseudo_labels.append(pseudo_label)
                    confident_images.append(images)
        
        return confident_images, pseudo_labels


class KnowledgeDistillation:
    """Knowledge distillation from a larger teacher model."""
    
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
    def distillation_loss(self, student_logits, teacher_logits, targets, criterion):
        """Compute distillation loss."""
        # Soft targets from teacher
        soft_targets = torch.sigmoid(teacher_logits / self.temperature)
        soft_prob = torch.sigmoid(student_logits / self.temperature)
        
        # KL divergence loss
        soft_loss = F.binary_cross_entropy(soft_prob, soft_targets, reduction='mean')
        soft_loss *= (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = criterion(student_logits, targets)
        
        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class GradualUnfreezing:
    """Gradual unfreezing strategy for fine-tuning."""
    
    def __init__(self, model, unfreeze_schedule):
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule  # Dict: {epoch: layers_to_unfreeze}
        
    def unfreeze_layers(self, epoch):
        """Unfreeze layers based on schedule."""
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule[epoch]
            for layer_name in layers_to_unfreeze:
                layer = getattr(self.model, layer_name, None)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
                    print(f"Unfroze layer: {layer_name}")


class CyclicLearningRate:
    """Cyclic learning rate scheduler."""
    
    def __init__(self, optimizer, base_lr=1e-5, max_lr=1e-3, step_size=2000, mode='triangular'):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.iteration = 0
        
    def step(self):
        """Update learning rate."""
        cycle = np.floor(1 + self.iteration / (2 * self.step_size))
        x = np.abs(self.iteration / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = 0.99999 ** self.iteration
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.iteration += 1


def create_advanced_config():
    """Create advanced configuration with all improvements."""
    return {
        # Model improvements
        'use_deep_supervision': True,
        'use_attention_gates': True,
        'use_cbam': True,
        'use_transformer_bottleneck': True,
        'dropout_prob': 0.1,
        
        # Training improvements
        'use_ema': True,
        'ema_decay': 0.9999,
        'use_gradient_clipping': True,
        'grad_clip_value': 1.0,
        'use_label_smoothing': True,
        'label_smoothing': 0.1,
        
        # Loss improvements
        'use_focal_loss': True,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'use_boundary_loss': True,
        'boundary_weight': 0.1,
        
        # Augmentation improvements
        'use_mixup': True,
        'mixup_alpha': 0.4,
        'use_cutmix': True,
        'cutmix_alpha': 1.0,
        'use_advanced_augmentation': True,
        
        # Post-processing
        'use_tta': True,
        'tta_num_augments': 8,
        'use_crf_postprocess': True,
        'remove_small_objects': True,
        'min_object_size': 100,
        
        # Ensemble
        'use_ensemble': False,  # Set to True if you have multiple models
        'ensemble_weights': None,
        
        # Semi-supervised
        'use_self_training': False,
        'confidence_threshold': 0.9,
        
        # Learning rate
        'use_cyclic_lr': False,
        'cyclic_base_lr': 1e-5,
        'cyclic_max_lr': 1e-3,
        
        # Gradual unfreezing
        'use_gradual_unfreezing': True,
        'unfreeze_schedule': {
            10: ['encoder4'],
            20: ['encoder3'],
            30: ['encoder2'],
            40: ['encoder1']
        }
    }


__all__ = [
    'TestTimeAugmentation', 'PostProcessing', 'EnsembleModel', 
    'AdversarialTraining', 'FocalLoss2D', 'SelfTraining',
    'KnowledgeDistillation', 'GradualUnfreezing', 'CyclicLearningRate',
    'create_advanced_config'
]