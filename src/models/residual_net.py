"""A lightweight neural residual appearance model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class ResidualNetOutput:
    """Outputs from the residual appearance network."""

    residual: torch.Tensor
    refined: torch.Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResidualAppearanceNet(nn.Module):
    """A compact U-Net that predicts a residual correction over physical rendering."""

    def __init__(
        self,
        in_channels: int = 10,
        base_channels: int = 32,
        out_channels: int = 3,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        self.clamp_output = clamp_output

        self.stem = ConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(
        self,
        physical_image: torch.Tensor,
        *,
        albedo: torch.Tensor | None = None,
        normals: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
        extra_features: torch.Tensor | None = None,
    ) -> ResidualNetOutput:
        features = [physical_image]
        if albedo is not None:
            features.append(albedo)
        if normals is not None:
            features.append(normals)
        if depth is not None:
            features.append(depth)
        if extra_features is not None:
            features.append(extra_features)

        x = torch.cat(features, dim=1)
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.bottleneck(x2)
        x = self.up2(x3, x1)
        x = self.up1(x, x0)
        residual = self.head(x)
        if self.clamp_output:
            refined = (physical_image + residual).clamp(0.0, 1.0)
        else:
            refined = physical_image + residual
        return ResidualNetOutput(residual=residual, refined=refined)
