from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class M00EncoderToMatch(nn.Module):
    def __init__(
        self,
        out_ch: int = 18,
        log_input: bool = False,
        log_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if log_input:
            raise ValueError("The published protocol uses untransformed M00 input.")
        del log_eps
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1),
        )

    def forward(self, m00: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        features = self.net(m00)
        return F.interpolate(
            features, size=target_hw, mode="bilinear", align_corners=False
        )


class FuseGate(nn.Module):
    def __init__(
        self,
        ch: int = 18,
        *,
        channels: int | None = None,
    ) -> None:
        super().__init__()
        channels = ch if channels is None else channels
        self.to_gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(
        self,
        hrnet_high: torch.Tensor,
        m00_features: torch.Tensor | None,
        m00_present: torch.Tensor,
    ) -> torch.Tensor:
        if m00_features is None:
            return hrnet_high
        gate = self.to_gate(m00_features) * m00_present
        return hrnet_high * (1.0 + gate)
