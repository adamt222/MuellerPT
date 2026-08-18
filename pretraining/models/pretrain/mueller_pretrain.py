from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from models.heads import HRNetDecompHead
from models.masking import apply_channel_mask
from models.spectral import FuseGate, M00EncoderToMatch


def _gradient_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(prediction)
    mask = mask.to(dtype=prediction.dtype)

    mask_x = mask[..., 1:] * mask[..., :-1]
    prediction_x = prediction[..., 1:] - prediction[..., :-1]
    target_x = target[..., 1:] - target[..., :-1]
    loss_x = ((prediction_x - target_x).abs() * mask_x).sum()
    loss_x = loss_x / mask_x.sum().clamp_min(1.0)

    mask_y = mask[..., 1:, :] * mask[..., :-1, :]
    prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    loss_y = ((prediction_y - target_y).abs() * mask_y).sum()
    loss_y = loss_y / mask_y.sum().clamp_min(1.0)
    return loss_x + loss_y


class MuellerPretrainModel(nn.Module):
    """Published decomposition-only MuellerPT objective."""

    def __init__(
        self,
        encoder: nn.Module,
        decomp_decoder: HRNetDecompHead,
        decomp_order: tuple[str, ...],
        m00_drop_prob: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decomp_decoder = decomp_decoder
        self.decomp_order = decomp_order
        self.m00_drop_prob = m00_drop_prob
        self.mask_token = nn.Parameter(torch.zeros(1, 16, 1, 1))
        channels = int(getattr(encoder, "highres_channels", 18))
        self.m00_enc = M00EncoderToMatch(out_ch=channels)
        self.fuse_gate = FuseGate(channels=channels)

    def _target_tensor(
        self, targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        missing = [name for name in self.decomp_order if name not in targets]
        if missing:
            raise KeyError(f"Missing decomposition targets: {missing}")
        return torch.cat([targets[name] for name in self.decomp_order], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
        m00: torch.Tensor,
        m00_present: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        x_corrupt = apply_channel_mask(x, channel_mask, self.mask_token)
        features = self.encoder(x_corrupt)
        high_resolution = features["hrnet_high"]
        if not torch.is_tensor(high_resolution):
            raise TypeError("HRNet high-resolution features must be a tensor.")

        if m00.ndim != 4 or m00.shape[1] != 1:
            raise ValueError(f"Expected M00 shape [B,1,H,W], got {tuple(m00.shape)}")
        present_map = m00_present.view(batch_size, 1, 1, 1).to(
            device=high_resolution.device,
            dtype=high_resolution.dtype,
        )
        if (
            self.training
            and self.m00_drop_prob > 0
            and torch.rand((), device=high_resolution.device) < self.m00_drop_prob
        ):
            m00_features = None
            present_map = torch.zeros_like(present_map)
        else:
            m00_features = self.m00_enc(
                m00.to(dtype=high_resolution.dtype),
                (high_resolution.shape[-2], high_resolution.shape[-1]),
            )

        fused = self.fuse_gate(high_resolution, m00_features, present_map)
        features = dict(features)
        features["hrnet_high"] = fused
        branches = list(features["hrnet_branches"])
        branches[0] = fused
        features["hrnet_branches"] = branches

        prediction = self.decomp_decoder(features, output_size=x.shape[-2:])
        target = self._target_tensor(targets)
        difference = F.smooth_l1_loss(
            prediction, target, reduction="none", beta=0.1
        )
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.to(device=x.device, dtype=difference.dtype)
        expanded_mask = valid_mask.expand_as(difference)
        decomposition_loss = (difference * expanded_mask).sum()
        decomposition_loss = decomposition_loss / expanded_mask.sum().clamp_min(1.0)
        decomposition_loss = decomposition_loss + 0.2 * _gradient_l1(
            prediction, target, valid_mask
        )
        return {"loss": decomposition_loss, "loss_decomp": decomposition_loss}
