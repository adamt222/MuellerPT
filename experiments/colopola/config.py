from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(os.environ.get("MUELLERPT_INPUT_ROOT", str(REPO_ROOT / "data")))
OUTPUT_ROOT = Path(os.environ.get("MUELLERPT_OUTPUT_ROOT", str(REPO_ROOT / "outputs")))
DEFAULT_CHECKPOINT = Path(os.environ.get("MUELLERPT_CHECKPOINT", str(OUTPUT_ROOT / "checkpoints" / "muellerpt_hrnet_w18_epoch150.pt")))


@dataclass
class TrainConfig:
    dataset_root: Path = INPUT_ROOT / "filtered_matrices" / "colopola"
    output_dir: Path = OUTPUT_ROOT / "results" / "colopola"
    run_name: str | None = None

    batch_size: int = 6
    num_workers: int = 4
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_max_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    split_seed: int = 42
    split_seeds: list[int] | None = None
    deterministic: bool = False

    rotation_deg: float = 30.0
    rotation_prob: float = 0.2
    flip_prob: float = 0.2
    center_crop_size: int | None = 600
    pad_multiple: int | None = 16

    num_classes: int = 2
    encoder_variant: str = "hrnet_w18"
    encoder_checkpoint: str | None = str(DEFAULT_CHECKPOINT)
    few_shot_percentages: list[int] = field(
        default_factory=lambda: [1, 5, 25, 50, 100]
    )

    early_stop_patience: int = 12
    early_stop_min_delta: float = 0.0


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
