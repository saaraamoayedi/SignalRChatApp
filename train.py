"""
Advanced training script for medical image segmentation.
Includes various optimizations and techniques to boost performance beyond 84% IoU.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import random
from tqdm import tqdm
import logging
from pathlib import Path

# Import our modules
from model import ImprovedUNet
from losses import CombinedLoss, UltimateLoss, AdaptiveLoss
from dataset import create_data_loaders, mix_augment


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration."""
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def calculate_metrics(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate IoU, Dice, and other metrics."""
    pred_binary = (pred > threshold).float()
    target = target.float()
    
    # IoU
    intersection = torch.sum(pred_binary * target, dim=[1, 2, 3])
    union = torch.sum(pred_binary + target, dim=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    # Dice
    dice = (2.0 * intersection + smooth) / (torch.sum(pred_binary, dim=[1, 2, 3]) + torch.sum(target, dim=[1, 2, 3]) + smooth)
    
    # Precision & Recall
    tp = intersection
    fp = torch.sum(pred_binary, dim=[1, 2, 3]) - tp
    fn = torch.sum(target, dim=[1, 2, 3]) - tp
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return {
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item()
    }


class EarlyStopping:
    """Early stopping utility."""
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class ModelEMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AdvancedTrainer:
    """Advanced trainer with multiple optimization techniques."""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.logger = setup_logging()
        
        # Setup loss function
        if config['loss_type'] == 'combined':
            self.criterion = CombinedLoss(pos_weight=config['pos_weight'])
        elif config['loss_type'] == 'ultimate':
            self.criterion = UltimateLoss(pos_weight=config['pos_weight'])
        elif config['loss_type'] == 'adaptive':
            self.criterion = AdaptiveLoss(pos_weight=config['pos_weight'])
        else:
            self.criterion = CombinedLoss(pos_weight=config['pos_weight'])
        
        # Setup optimizer
        if config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(), 
                lr=config['lr'], 
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(), 
                lr=config['lr'], 
                momentum=0.9, 
                weight_decay=config['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=config['lr'], 
                weight_decay=config['weight_decay']
            )
        
        # Setup scheduler
        if config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        elif config['scheduler'] == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=config['lr'], 
                epochs=config['num_epochs'], 
                steps_per_epoch=len(train_loader)
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        
        # Setup EMA and Early Stopping
        self.ema = ModelEMA(model, decay=config.get('ema_decay', 0.9999)) if config.get('use_ema', False) else None
        self.early_stopping = EarlyStopping(patience=config['patience'])
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        running_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Apply augmentations
            if self.config.get('use_mix_augment', True):
                images, masks = mix_augment(images, masks, alpha=0.4)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Update EMA
            if self.ema:
                self.ema.update()
            
            # Calculate metrics
            with torch.no_grad():
                outputs_sigmoid = torch.sigmoid(outputs)
                metrics = calculate_metrics(outputs_sigmoid, masks)
            
            # Update running statistics
            running_loss += loss.item()
            for key in running_metrics:
                running_metrics[key] += metrics[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{metrics["iou"]:.4f}',
                'dice': f'{metrics["dice"]:.4f}'
            })
            
            # Update scheduler if OneCycleLR
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
        
        # Calculate epoch averages
        epoch_loss = running_loss / len(self.train_loader)
        epoch_metrics = {k: v / len(self.train_loader) for k, v in running_metrics.items()}
        
        return epoch_loss, epoch_metrics
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        running_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Apply EMA for validation
        if self.ema:
            self.ema.apply_shadow()
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc='Validation'):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                outputs_sigmoid = torch.sigmoid(outputs)
                metrics = calculate_metrics(outputs_sigmoid, masks)
                
                running_loss += loss.item()
                for key in running_metrics:
                    running_metrics[key] += metrics[key]
        
        # Restore original weights
        if self.ema:
            self.ema.restore()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_metrics = {k: v / len(self.val_loader) for k, v in running_metrics.items()}
        
        return epoch_loss, epoch_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.ema:
            checkpoint['ema_state_dict'] = self.ema.shadow
        
        # Save latest checkpoint
        torch.save(checkpoint, 'last_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, 'best_checkpoint.pth')
            self.logger.info(f'New best model saved at epoch {epoch+1}')
    
    def plot_progress(self, epoch):
        """Plot training progress."""
        epochs_range = range(1, len(self.train_losses) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(epochs_range, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs_range, self.val_losses, 'r-', label='Val Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # IoU
        train_ious = [m['iou'] for m in self.train_metrics]
        val_ious = [m['iou'] for m in self.val_metrics]
        ax2.plot(epochs_range, train_ious, 'b-', label='Train IoU')
        ax2.plot(epochs_range, val_ious, 'r-', label='Val IoU')
        ax2.set_title('IoU')
        ax2.legend()
        ax2.grid(True)
        
        # Dice
        train_dice = [m['dice'] for m in self.train_metrics]
        val_dice = [m['dice'] for m in self.val_metrics]
        ax3.plot(epochs_range, train_dice, 'b-', label='Train Dice')
        ax3.plot(epochs_range, val_dice, 'r-', label='Val Dice')
        ax3.set_title('Dice Score')
        ax3.legend()
        ax3.grid(True)
        
        # Precision & Recall
        train_prec = [m['precision'] for m in self.train_metrics]
        val_prec = [m['precision'] for m in self.val_metrics]
        train_rec = [m['recall'] for m in self.train_metrics]
        val_rec = [m['recall'] for m in self.val_metrics]
        ax4.plot(epochs_range, train_prec, 'b-', label='Train Precision')
        ax4.plot(epochs_range, val_prec, 'r-', label='Val Precision')
        ax4.plot(epochs_range, train_rec, 'g-', label='Train Recall')
        ax4.plot(epochs_range, val_rec, 'orange', label='Val Recall')
        ax4.set_title('Precision & Recall')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_progress_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Update scheduler
            if not isinstance(self.scheduler, OneCycleLR):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log progress
            self.logger.info(
                f"Epoch [{epoch+1}/{self.config['num_epochs']}] "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}, "
                f"Train Dice: {train_metrics['dice']:.4f}, Val Dice: {val_metrics['dice']:.4f}"
            )
            
            # Save best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            
            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_progress(epoch)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Final plot
        self.plot_progress(epoch)


def main():
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(42)
    
    # Configuration
    config = {
        # Data
        'batch_size': 8,
        'image_size': (384, 384),
        'num_workers': 4,
        
        # Model
        'num_classes': 1,
        'pretrained': True,
        'freeze_encoder': False,
        'dropout_prob': 0.1,
        'use_cbam': True,
        'use_se': False,
        'deep_supervision': False,
        
        # Training
        'num_epochs': 100,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',  # 'adam', 'adamw', 'sgd'
        'scheduler': 'cosine',  # 'plateau', 'cosine', 'onecycle'
        'grad_clip': 1.0,
        
        # Loss
        'loss_type': 'ultimate',  # 'combined', 'ultimate', 'adaptive'
        'pos_weight': 5.0,
        
        # Augmentation
        'use_mix_augment': True,
        'use_advanced_augmentation': True,
        
        # Optimization
        'use_ema': True,
        'ema_decay': 0.9999,
        'patience': 15,
    }
    
    # Data paths (update these to your actual paths)
    train_image_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/train/images'
    train_mask_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/train/masks'
    test_image_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/test/images'
    test_mask_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/test/masks'
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_image_path, train_mask_path,
        test_image_path, test_mask_path,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers'],
        use_advanced_augmentation=config['use_advanced_augmentation']
    )
    
    # Create model
    model = ImprovedUNet(
        num_classes=config['num_classes'],
        pretrained=config['pretrained'],
        freeze_encoder=config['freeze_encoder'],
        dropout_prob=config['dropout_prob'],
        cbam=config['use_cbam'],
        use_se=config['use_se'],
        deep_supervision=config['deep_supervision']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer and start training
    trainer = AdvancedTrainer(model, train_loader, val_loader, device, config)
    trainer.train()


if __name__ == "__main__":
    main()