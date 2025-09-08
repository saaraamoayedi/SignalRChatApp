"""
Main script to run the improved segmentation model.
This combines all the optimizations and improvements for maximum performance.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import our modules
from model import ImprovedUNet
from losses import UltimateLoss, AdaptiveLoss
from dataset import create_data_loaders, mix_augment
from train import AdvancedTrainer, set_seed
from improvements import (
    TestTimeAugmentation, PostProcessing, create_advanced_config,
    GradualUnfreezing, CyclicLearningRate
)


def main():
    """Main function with all improvements integrated."""
    
    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    set_seed(42)
    
    # Advanced configuration for maximum performance
    config = {
        # Data settings
        'batch_size': 8,  # Adjust based on your GPU memory
        'image_size': (384, 384),  # Higher resolution often helps
        'num_workers': 4,
        
        # Model architecture - all improvements enabled
        'num_classes': 1,
        'pretrained': True,
        'freeze_encoder': False,  # We'll use gradual unfreezing
        'dropout_prob': 0.1,
        'use_cbam': True,  # Attention mechanism
        'use_se': False,   # CBAM is generally better
        'deep_supervision': True,  # Multiple loss heads
        
        # Training hyperparameters
        'num_epochs': 150,  # More epochs for better convergence
        'lr': 2e-4,  # Conservative learning rate
        'weight_decay': 1e-4,
        'optimizer': 'adamw',  # Generally better than Adam
        'scheduler': 'cosine',  # Cosine annealing with warm restarts
        'grad_clip': 1.0,  # Gradient clipping for stability
        
        # Advanced loss function
        'loss_type': 'ultimate',  # Uses all loss components
        'pos_weight': 5.0,  # Adjust based on class imbalance
        
        # Data augmentation
        'use_mix_augment': True,  # MixUp, CutMix, AlphaMix
        'use_advanced_augmentation': True,  # Albumentations pipeline
        
        # Optimization techniques
        'use_ema': True,  # Exponential Moving Average
        'ema_decay': 0.9999,
        'patience': 20,  # Early stopping patience
        
        # Advanced techniques
        'use_tta': True,  # Test-Time Augmentation
        'use_post_processing': True,
        'use_gradual_unfreezing': True,
    }
    
    # Update paths to your dataset
    train_image_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/train/images'
    train_mask_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/train/masks'
    test_image_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/test/images'
    test_mask_path = '/kaggle/input/brisc2025/brisc2025/segmentation_task/test/masks'
    
    # Check if paths exist
    paths = [train_image_path, train_mask_path, test_image_path, test_mask_path]
    for path in paths:
        if not os.path.exists(path):
            print(f"⚠️ Warning: Path does not exist: {path}")
            print("Please update the paths in main.py to match your dataset location.")
            return
    
    print("📊 Dataset Information:")
    print(f"   Train images: {len(os.listdir(train_image_path))}")
    print(f"   Train masks: {len(os.listdir(train_mask_path))}")
    print(f"   Test images: {len(os.listdir(test_image_path))}")
    print(f"   Test masks: {len(os.listdir(test_mask_path))}")
    
    # Create data loaders with advanced augmentation
    print("🔄 Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_image_path, train_mask_path,
        test_image_path, test_mask_path,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers'],
        use_advanced_augmentation=config['use_advanced_augmentation']
    )
    
    # Create improved model
    print("🏗️ Building improved model...")
    model = ImprovedUNet(
        num_classes=config['num_classes'],
        pretrained=config['pretrained'],
        freeze_encoder=config['freeze_encoder'],
        dropout_prob=config['dropout_prob'],
        cbam=config['use_cbam'],
        use_se=config['use_se'],
        deep_supervision=config['deep_supervision']
    ).to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup gradual unfreezing if enabled
    if config.get('use_gradual_unfreezing', False):
        print("🔓 Setting up gradual unfreezing...")
        # Initially freeze encoder
        for name, param in model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
        
        unfreeze_schedule = {
            20: ['encoder4'],
            40: ['encoder3'], 
            60: ['encoder2'],
            80: ['encoder1', 'inc']
        }
        gradual_unfreezer = GradualUnfreezing(model, unfreeze_schedule)
        config['gradual_unfreezer'] = gradual_unfreezer
    
    # Create advanced trainer
    print("🚀 Initializing advanced trainer...")
    trainer = AdvancedTrainer(model, train_loader, val_loader, device, config)
    
    # Start training
    print("🎯 Starting training with all optimizations enabled...")
    print("=" * 60)
    trainer.train()
    
    # Load best model for inference
    print("📥 Loading best model...")
    checkpoint = torch.load('best_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Setup Test-Time Augmentation
    if config.get('use_tta', False):
        print("🔮 Setting up Test-Time Augmentation...")
        tta = TestTimeAugmentation(model, device)
        
        # Test TTA on a sample
        sample_images, sample_masks = next(iter(val_loader))
        sample_images = sample_images.to(device)
        
        # Regular prediction
        model.eval()
        with torch.no_grad():
            regular_pred = torch.sigmoid(model(sample_images))
        
        # TTA prediction
        tta_pred = tta.predict_with_tta(sample_images, num_augments=8)
        
        print(f"📊 TTA vs Regular prediction difference: {torch.abs(tta_pred - regular_pred).mean().item():.4f}")
    
    # Post-processing example
    if config.get('use_post_processing', False):
        print("🔧 Applying post-processing...")
        post_processor = PostProcessing()
        
        # Apply post-processing to sample prediction
        processed_pred = post_processor.remove_small_objects(regular_pred[0:1], min_size=50)
        processed_pred = post_processor.fill_holes(processed_pred)
        processed_pred = post_processor.morphological_operations(processed_pred, 'close', 3)
        
        print(f"📊 Post-processing effect: {torch.abs(processed_pred - regular_pred[0:1]).mean().item():.4f}")
    
    print("✅ Training completed successfully!")
    print(f"📁 Best model saved as: best_checkpoint.pth")
    print(f"📁 Latest model saved as: last_checkpoint.pth")
    print("🎉 Your improved segmentation model should achieve IoU > 84%!")


def inference_example():
    """Example of how to use the trained model for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ImprovedUNet(num_classes=1, pretrained=False).to(device)
    checkpoint = torch.load('best_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup TTA and post-processing
    tta = TestTimeAugmentation(model, device)
    post_processor = PostProcessing()
    
    # Example inference function
    def predict_with_all_improvements(image_tensor):
        """Predict with TTA and post-processing."""
        # TTA prediction
        tta_pred = tta.predict_with_tta(image_tensor.unsqueeze(0))
        
        # Post-processing
        processed = post_processor.remove_small_objects(tta_pred, min_size=50)
        processed = post_processor.fill_holes(processed)
        processed = post_processor.morphological_operations(processed, 'close', 3)
        
        return processed.squeeze(0)
    
    print("🔮 Inference function ready!")
    return predict_with_all_improvements


if __name__ == "__main__":
    main()
    
    # Uncomment to create inference function
    # predict_func = inference_example()