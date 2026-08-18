from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAINING_ROOT = REPO_ROOT / "pretraining"
if str(PRETRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAINING_ROOT))

_hrnet_import_error: Optional[Exception] = None
try:
    from models.encoders import HRNetEncoder
except Exception as exc:
    _hrnet_import_error = exc
    HRNetEncoder = None
try:
    from models.spectral import FuseGate, M00EncoderToMatch
except Exception:
    FuseGate = None
    M00EncoderToMatch = None


class HRNetSegHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: str = "bilinear",
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.upsample = upsample
        mid = mid_channels or in_channels
        if upsample == "deconv":
            self.upsample_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, mid, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(mid, mid, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            )
            self.refine = nn.Sequential(
                nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(mid, out_channels, kernel_size=1)
        elif upsample == "bilinear":
            self.upsample_block = None
            self.refine = nn.Sequential(
                nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(mid, out_channels, kernel_size=1)
        else:
            raise ValueError(f"Unsupported upsample mode: {upsample}")

    def _select_highres(self, feats) -> torch.Tensor:
        if isinstance(feats, dict):
            if "hrnet_high" in feats:
                return feats["hrnet_high"]
            if "hrnet_branches" in feats and feats["hrnet_branches"]:
                return feats["hrnet_branches"][0]
        if isinstance(feats, (list, tuple)) and len(feats) > 0:
            return feats[0]
        if torch.is_tensor(feats):
            return feats
        raise ValueError("Unsupported HRNet features input for segmentation head.")

    def forward(self, feats, output_size: Optional[tuple[int, int]] = None) -> torch.Tensor:
        x = self._select_highres(feats)
        if self.upsample_block is None:
            if output_size is None:
                x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
            else:
                x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
            x = self.refine(x)
            return self.head(x)
        x = self.upsample_block(x)
        if output_size is not None and x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        x = self.refine(x)
        return self.head(x)


class HRNetSegmentation(nn.Module):
    def __init__(
        self,
        variant: str = "hrnet_w18",
        in_channels: int = 16,
        num_classes: int = 2,
        pretrained: bool = False,
        head_upsample: str = "bilinear",
        use_m00: bool = False,
        m00_log_input: bool = False,
    ) -> None:
        super().__init__()
        if HRNetEncoder is None:
            detail = (
                f" ({type(_hrnet_import_error).__name__}: {_hrnet_import_error})"
                if _hrnet_import_error is not None
                else ""
            )
            raise ImportError(
                "HRNetEncoder is unavailable; check the bundled pretraining "
                f"package and repository layout{detail}."
            )
        self.encoder = HRNetEncoder(
            variant=variant,
            in_chans=in_channels,
            pretrained=pretrained,
        )
        in_ch = getattr(self.encoder, "highres_channels", None)
        if in_ch is None:
            raise ValueError("HRNetEncoder missing highres_channels.")
        self.head = HRNetSegHead(
            in_channels=in_ch,
            out_channels=num_classes,
            upsample=head_upsample,
        )
        self.use_m00 = bool(use_m00)
        if self.use_m00:
            if M00EncoderToMatch is None or FuseGate is None:
                raise ImportError("M00EncoderToMatch/FuseGate not available.")
            self.m00_enc = M00EncoderToMatch(out_ch=in_ch, log_input=m00_log_input)
            self.fuse_gate = FuseGate(ch=in_ch)
        else:
            self.m00_enc = None
            self.fuse_gate = None

    def forward(
        self,
        x: torch.Tensor,
        m00: Optional[torch.Tensor] = None,
        m00_present: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = self.encoder(x)
        if self.use_m00:
            if not isinstance(feats, dict) or "hrnet_high" not in feats:
                raise ValueError("M00 fusion requires encoder features with key 'hrnet_high'.")
            hrnet_high = feats["hrnet_high"]
            target_hw = (hrnet_high.size(-2), hrnet_high.size(-1))
            b = hrnet_high.shape[0]

            if m00 is not None:
                if m00.ndim == 3:
                    m00 = m00.unsqueeze(1)
                if m00.ndim != 4 or m00.shape[1] != 1:
                    raise ValueError(
                        f"m00 must have shape [B,1,H,W] or [B,H,W], got {tuple(m00.shape)}."
                    )
                m00 = m00.to(device=hrnet_high.device, dtype=hrnet_high.dtype)

            if m00_present is None:
                if m00 is None:
                    m00_present_map = hrnet_high.new_zeros((b, 1, 1, 1))
                else:
                    m00_present_map = hrnet_high.new_ones((b, 1, 1, 1))
            else:
                if m00_present.ndim == 1:
                    m00_present = m00_present.view(b, 1, 1, 1)
                elif m00_present.ndim == 2:
                    m00_present = m00_present.view(b, 1, 1, 1)
                elif m00_present.ndim == 4 and m00_present.shape[1:] == (1, 1, 1):
                    pass
                else:
                    raise ValueError(
                        "m00_present must have shape [B], [B,1], or [B,1,1,1]."
                    )
                m00_present_map = m00_present.to(device=hrnet_high.device, dtype=hrnet_high.dtype)

            f_m00 = None if m00 is None else self.m00_enc(m00, target_hw)
            hrnet_high = self.fuse_gate(hrnet_high, f_m00, m00_present_map)
            feats = dict(feats)
            feats["hrnet_high"] = hrnet_high
            if "hrnet_branches" in feats and isinstance(feats["hrnet_branches"], (list, tuple)):
                branches = list(feats["hrnet_branches"])
                if branches:
                    branches[0] = hrnet_high
                    feats["hrnet_branches"] = branches
        return self.head(feats, output_size=x.shape[-2:])
