from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class HRNetDecompHead(nn.Module):
    """Bilinear decoder used by the published MuellerPT checkpoint."""

    def __init__(self, in_channels: int, decomp_order: tuple[str, ...]) -> None:
        super().__init__()
        self.decomp_order = decomp_order
        self.head = nn.Conv2d(in_channels, len(decomp_order), kernel_size=1)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        x = features["hrnet_high"]
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        parts = list(torch.split(self.head(x), 1, dim=1))
        for index, name in enumerate(self.decomp_order):
            if name == "retardance":
                parts[index] = math.pi * torch.sigmoid(parts[index])
            else:
                parts[index] = torch.sigmoid(parts[index])
        return torch.cat(parts, dim=1)
