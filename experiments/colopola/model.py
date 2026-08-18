from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

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


class MILAttentionPool(nn.Module):
    """
    Single-head MIL attention pooling over spatial locations.
    Treat each HxW position as an instance and compute a weighted bag embedding.
    """

    def __init__(self, in_dim: int, attn_hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        instances = x.view(b, c, h * w).transpose(1, 2)  # [B, N, C], N=H*W
        scores = self.attn(instances).squeeze(-1)  # [B, N]
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), instances).squeeze(1)  # [B, C]
        return pooled


class HRNetClassifier(nn.Module):
    def __init__(
        self,
        variant: str = "hrnet_w18",
        in_channels: int = 16,
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if HRNetEncoder is None:
            detail = (
                f" ({type(_hrnet_import_error).__name__}: {_hrnet_import_error})"
                if _hrnet_import_error is not None
                else ""
            )
            raise ImportError(f"HRNetEncoder not available; check pretraining path setup{detail}.")

        self.encoder = HRNetEncoder(
            variant=variant,
            in_chans=in_channels,
            pretrained=pretrained,
        )
        in_ch = getattr(self.encoder, "highres_channels", None)
        if in_ch is None:
            raise ValueError("HRNetEncoder missing highres_channels.")

        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = MILAttentionPool(in_dim=in_ch, attn_hidden_dim=max(32, in_ch))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if isinstance(feats, dict) and "hrnet_high" in feats:
            high = feats["hrnet_high"]
        elif isinstance(feats, (list, tuple)) and len(feats) > 0:
            high = feats[0]
        elif torch.is_tensor(feats):
            high = feats
        else:
            raise ValueError("Unsupported HRNet features for classifier head.")
        refined = self.refine(high)
        pooled = self.pool(refined)
        pooled = self.dropout(pooled)
        return self.head(pooled)


def load_encoder_from_checkpoint(model: HRNetClassifier, ckpt_path: Path) -> tuple[int, int, int]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    # Some checkpoints store encoder-only weights under an "encoder" key.
    if isinstance(state, dict) and "encoder" in state and isinstance(state["encoder"], dict):
        state = state["encoder"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")

    enc_state = model.encoder.state_dict()
    filtered = {}

    for key, value in state.items():
        k = key
        for prefix in ("module.encoder.", "encoder.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
                break
        if k in enc_state:
            filtered[k] = value

    if not filtered:
        sample_keys = list(state.keys())[:8]
        raise ValueError(
            "No encoder weights were matched from checkpoint. "
            f"Checkpoint: {ckpt_path}. "
            f"Sample keys: {sample_keys}"
        )

    missing, unexpected = model.encoder.load_state_dict(filtered, strict=False)
    return len(filtered), len(missing), len(unexpected)
