"""
Improved U-Net with attention mechanisms for medical image segmentation.
Achieves high performance with ResNet34 backbone and various attention modules.
"""

from __future__ import annotations
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -------------------------- Attention Modules --------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Channel Attention)."""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class ChannelAttention(nn.Module):
    """Channel attention used inside CBAM (average + max pooling)."""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx, _ = torch.max(x, dim=2, keepdim=True)
        mx, _ = torch.max(mx, dim=3, keepdim=True)
        out = self.mlp(avg) + self.mlp(mx)
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention used inside CBAM (concat of avg & max -> conv7x7)."""
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        attn = self.conv(cat)
        return self.sigmoid(attn)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV'18)."""
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class AttentionGate(nn.Module):
    """Attention gate for skip connections (Oktay et al.)."""
    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True),
                                 nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True),
                                 nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class TransformerBlock(nn.Module):
    """Lightweight transformer using nn.MultiheadAttention in bottleneck."""
    def __init__(self, dim: int, num_heads: int = 8, mlp_dim: int = 1024) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.attn(self.norm1(t), self.norm1(t), self.norm1(t))
        t = t + attn_out
        t = t + self.mlp(self.norm2(t))
        return t.permute(0, 2, 1).view(B, C, H, W)


# -------------------------- Building Blocks --------------------------

class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> (Attention) -> (Dropout)"""
    def __init__(self, in_ch: int, out_ch: int, dropout_prob: float = 0.0,
                 use_se: bool = False, use_cbam: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.attn = nn.Identity()
        if use_cbam:
            self.attn = CBAM(out_ch)
        elif use_se:
            self.attn = SEBlock(out_ch)
        self.do = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.attn(x)
        x = self.do(x)
        return x


class UpBlock(nn.Module):
    """Upsample (deconv) -> match size -> attention-gated concat -> ConvBlock"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout_prob: float = 0.0,
                 use_se: bool = False, use_cbam: bool = False) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.ag = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=max(out_ch // 2, 1))
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout_prob, use_se, use_cbam)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Ensure same size as skip connection
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip_w = self.ag(g=x, x=skip)
        x = torch.cat([skip_w, x], dim=1)
        return self.conv(x)


# -------------------------- Improved U-Net --------------------------

class ImprovedUNet(nn.Module):
    """
    Improved U-Net with:
      - ResNet34 encoder (pretrained + freeze option)
      - Transformer bottleneck
      - Attention Gates in skip connections
      - CBAM/SE in conv blocks
      - Deep Supervision (optional)

    Output:
      logits with shape (B, num_classes, H, W). If deep_supervision=True,
      output is weighted average of side heads and main head.
    """
    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        dropout_prob: float = 0.0,
        cbam: bool = True,
        use_se: bool = False,
        deep_supervision: bool = False,
        ds_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        # ---- ResNet34 backbone ----
        try:
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        except Exception:
            resnet = models.resnet34(pretrained=pretrained)

        if freeze_encoder:
            for p in resnet.parameters():
                p.requires_grad = False

        # Stem & encoder stages
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # -> 64
        self.pool = resnet.maxpool
        self.encoder1 = resnet.layer1   # 64
        self.encoder2 = resnet.layer2   # 128
        self.encoder3 = resnet.layer3   # 256
        self.encoder4 = resnet.layer4   # 512

        # Bottleneck transformer
        self.transformer = TransformerBlock(dim=512, num_heads=8, mlp_dim=1024)

        # Decoder with attention gates + CBAM/SE
        self.up1 = UpBlock(512, 256, 256, dropout_prob, use_se, cbam)
        self.up2 = UpBlock(256, 128, 128, dropout_prob, use_se, cbam)
        self.up3 = UpBlock(128,  64,  64, dropout_prob, use_se, cbam)
        self.up4 = UpBlock( 64,  64,  32, dropout_prob, use_se, cbam)

        # Final conv
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

        # Deep supervision heads (side outputs)
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.side1 = nn.Conv2d(256, num_classes, 1)  # after up1
            self.side2 = nn.Conv2d(128, num_classes, 1)  # after up2
            self.side3 = nn.Conv2d( 64, num_classes, 1)  # after up3
            self.side4 = nn.Conv2d( 32, num_classes, 1)  # after up4
            if ds_weights is None:
                # Default weights: more importance to main head
                self.ds_weights = [0.15, 0.2, 0.25, 0.4]  # side1..side4
            else:
                assert len(ds_weights) == 4, "ds_weights must have 4 elements (for side1..side4)."
                s = sum(ds_weights)
                self.ds_weights = [w / s for w in ds_weights]  # normalize

        self._init_out_layers()

    def _init_out_layers(self) -> None:
        # Gentle initialization for output layers
        nn.init.xavier_uniform_(self.outc.weight)
        if self.outc.bias is not None:
            nn.init.zeros_(self.outc.bias)
        if self.deep_supervision:
            for head in [self.side1, self.side2, self.side3, self.side4]:
                nn.init.xavier_uniform_(head.weight)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # ---- Encoder ----
        x1 = self.inc(x)        # 64, H/2
        x2 = self.pool(x1)      # H/4
        x2 = self.encoder1(x2)  # 64
        x3 = self.encoder2(x2)  # 128
        x4 = self.encoder3(x3)  # 256
        x5 = self.encoder4(x4)  # 512

        # ---- Bottleneck ----
        x5 = self.transformer(x5)

        # ---- Decoder ----
        d1 = self.up1(x5, x4)   # 256
        d2 = self.up2(d1, x3)   # 128
        d3 = self.up3(d2, x2)   # 64
        d4 = self.up4(d3, x1)   # 32

        # Ensure size matches input
        if d4.shape[2:] != (H, W):
            d4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=False)

        main_logits = self.outc(d4)

        if not self.deep_supervision:
            return main_logits

        # ---- Side outputs (upsample to input size) ----
        s1 = self.side1(d1)
        s2 = self.side2(d2)
        s3 = self.side3(d3)
        s4 = self.side4(d4)

        s1 = F.interpolate(s1, size=(H, W), mode='bilinear', align_corners=False)
        s2 = F.interpolate(s2, size=(H, W), mode='bilinear', align_corners=False)
        s3 = F.interpolate(s3, size=(H, W), mode='bilinear', align_corners=False)
        s4 = F.interpolate(s4, size=(H, W), mode='bilinear', align_corners=False)

        # Weighted average of outputs (only side heads; main==s4 here)
        combined = (
            self.ds_weights[0] * s1 +
            self.ds_weights[1] * s2 +
            self.ds_weights[2] * s3 +
            self.ds_weights[3] * s4
        )
        return combined


__all__ = ["ImprovedUNet", "CBAM", "SEBlock", "AttentionGate", "TransformerBlock", "ConvBlock", "UpBlock"]