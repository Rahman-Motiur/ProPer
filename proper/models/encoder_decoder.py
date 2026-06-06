from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DistributionHead(nn.Module):
    def __init__(self, channels: int, latent_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mu = nn.Linear(channels, latent_dim)
        self.logvar = nn.Linear(channels, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(x).flatten(1)
        return self.mu(pooled), self.logvar(pooled).clamp(-8.0, 8.0)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, latent_dim: int):
        super().__init__()
        self.stem = ConvBlock(in_channels, base_channels)
        self.down1 = ConvBlock(base_channels, base_channels * 2)
        self.down2 = ConvBlock(base_channels * 2, base_channels * 4)
        self.dist = DistributionHead(base_channels * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = F.max_pool2d(x, 2)
        x = self.down1(x)
        x = F.max_pool2d(x, 2)
        features = self.down2(x)
        mu, logvar = self.dist(features)
        return features, mu, logvar


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, latent_dim: int):
        super().__init__()
        self.up1 = ConvBlock(in_channels, out_channels * 2)
        self.up2 = ConvBlock(out_channels * 2, out_channels)
        self.dist = DistributionHead(out_channels, latent_dim)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        features = self.up2(x)
        mu, logvar = self.dist(features)
        return features, mu, logvar
