from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("MUELLERPT_OUTPUT_ROOT", str(REPO_ROOT / "outputs")))
INPUT_ROOT = Path(os.environ.get("MUELLERPT_INPUT_ROOT", str(REPO_ROOT / "data")))
POLAMBRIMETRY_ROOT = Path(os.environ.get("MUELLERPT_POLAMBRIMETRY_ROOT", str(INPUT_ROOT / "polambrimetry")))
DEFAULT_SSL_CHECKPOINT = Path(os.environ.get("MUELLERPT_CHECKPOINT", str(OUTPUT_ROOT / "checkpoints" / "muellerpt_hrnet_w18_epoch150.pt")))


C_MAX = 16
PATCH_SIZE = 64 


@dataclass
class TrainConfig:
    dataset_root: Path = POLAMBRIMETRY_ROOT
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    folds: int = 6
    max_specimens: int | None = 6
    few_shot_percentages: List[int] = field(default_factory=lambda: [1, 5, 25, 50, 100])
    rotation_deg: float = 30
    rotation_prob: float = 0.8
    flip_prob: float = 0.8
    num_classes: int = 2
    encoder_variant: str = "hrnet_w18"
    seg_encoder_checkpoint: str | None = str(DEFAULT_SSL_CHECKPOINT)
    center_crop_size: int | None = 600
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 5
    lr_plateau_min_lr: float = 1e-6
    nested_early_stop_patience: int = 15
    early_stop_min_delta: float = 0.0
    seed: int = 400
    deterministic: bool = True
    m00_log_eps: float = 1e-6


GM_LABELS = {1, 2, 3, 4, 5, 7, 10, 11, 13, 18}
WM_LABELS = {6, 9, 12, 15, 16, 17, 25}
IGNORE_INDEX = 255


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
