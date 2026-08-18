from __future__ import annotations

from typing import Dict, List

import timm
import torch
import torch.nn.functional as F
from torch import nn


class HRNetEncoder(nn.Module):
    """Fixed 16-channel HRNet-W18 encoder used for MuellerPT pretraining."""

    highres_channels = 18
    highres_stride = 4

    def __init__(
        self,
        variant: str = "hrnet_w18",
        in_chans: int = 16,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if variant != "hrnet_w18" or in_chans != 16 or pretrained:
            raise ValueError(
                "The publication-focused encoder supports only "
                "variant='hrnet_w18', in_chans=16, pretrained=False."
            )
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            in_chans=in_chans,
        )
        if not hasattr(self.backbone, "stages"):
            raise ValueError("The timm HRNet-W18 model does not expose stages().")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        height, width = x.shape[-2:]
        pad_height = (32 - height % 32) % 32
        pad_width = (32 - width % 32) % 32
        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height), value=0)

        x = self.backbone.act1(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.act2(self.backbone.bn2(self.backbone.conv2(x)))
        branches = self.backbone.stages(x)
        if not isinstance(branches, (list, tuple)) or not branches:
            raise RuntimeError("HRNet stages() did not return branch features.")
        return {"hrnet_branches": branches, "hrnet_high": branches[0]}
