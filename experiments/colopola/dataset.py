from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAINING_ROOT = REPO_ROOT / "pretraining"
if PRETRAINING_ROOT.exists() and str(PRETRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAINING_ROOT))

from augmentations.mueller_rotation import RandomMuellerRotationCustom


@dataclass(frozen=True)
class ColopolaSample:
    path: Path
    label: int  # 0=normal, 1=cancer


def _label_from_name(path: Path) -> int:
    prefix = path.name.split("__", 1)[0].strip().lower()
    if prefix == "normal":
        return 0
    if prefix == "cancer":
        return 1
    raise ValueError(f"Unknown class in filename: {path.name}")


def load_colopola_samples(
    data_root: Path,
    max_samples: int | None = None,
) -> List[ColopolaSample]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Colopola root not found: {root}")

    all_samples = [
        ColopolaSample(path=p, label=_label_from_name(p))
        for p in sorted(root.glob("*_filtered.npy"))
    ]
    samples = all_samples
    if max_samples is not None and max_samples < len(all_samples):
        by_label: Dict[int, List[ColopolaSample]] = {0: [], 1: []}
        for sample in all_samples:
            by_label[sample.label].append(sample)

        per_label = max_samples // 2
        selected = by_label[0][:per_label] + by_label[1][:per_label]
        remaining = max_samples - len(selected)
        if remaining > 0:
            leftovers = by_label[0][per_label:] + by_label[1][per_label:]
            selected.extend(leftovers[:remaining])
        samples = selected
    if not samples:
        raise FileNotFoundError(f"No '*_filtered.npy' files under {root}")
    return samples


def _split_counts(
    n: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    n_test = n - n_train - n_val

    if n_test < 0:
        n_val = max(0, n_val + n_test)
        n_test = n - n_train - n_val
    if n_test < 0:
        n_train = max(0, n_train + n_test)
        n_test = n - n_train - n_val

    if n > 0 and test_fraction > 0 and n_test == 0:
        if n_train > 1:
            n_train -= 1
            n_test += 1
        elif n_val > 1:
            n_val -= 1
            n_test += 1

    if n > 0 and val_fraction > 0 and n_val == 0:
        if n_train > 1:
            n_train -= 1
            n_val += 1
        elif n_test > 1:
            n_test -= 1
            n_val += 1

    return n_train, n_val, n_test


def split_colopola_samples(
    samples: Iterable[ColopolaSample],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[List[ColopolaSample], List[ColopolaSample], List[ColopolaSample]]:
    if train_fraction < 0 or val_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions must be non-negative.")
    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total_fraction:.6f}."
        )

    by_label: Dict[int, List[ColopolaSample]] = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)

    train: List[ColopolaSample] = []
    val: List[ColopolaSample] = []
    test: List[ColopolaSample] = []

    for label, label_samples in by_label.items():
        rng = random.Random(seed + label)
        items = list(label_samples)
        rng.shuffle(items)

        n_train, n_val, _ = _split_counts(
            len(items), train_fraction, val_fraction, test_fraction
        )
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    rng_all = random.Random(seed)
    rng_all.shuffle(train)
    rng_all.shuffle(val)
    rng_all.shuffle(test)
    return train, val, test


class _ColopolaAugmenter:
    def __init__(
        self,
        rotation_prob: float = 0.8,
        rotation_deg: float = 30.0,
        flip_prob: float = 0.8,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotate = RandomMuellerRotationCustom(
            degrees=rotation_deg,
            p=rotation_prob,
            any=True,
            fill=0,
        )

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        frame = self.rotate(frame)
        if random.random() < self.flip_prob:
            frame = torch.flip(frame, dims=[2])
        if random.random() < self.flip_prob:
            frame = torch.flip(frame, dims=[1])
        return frame


class ColopolaClassificationDataset(Dataset):
    def __init__(
        self,
        samples: List[ColopolaSample],
        augment: bool = False,
        rotation_prob: float = 0.8,
        flip_prob: float = 0.8,
        rotation_deg: float = 30.0,
        center_crop_size: int | None = None,
        pad_multiple: int | None = 16,
    ) -> None:
        self.samples = list(samples)
        self.center_crop_size = center_crop_size
        self.pad_multiple = pad_multiple
        self.augmenter = (
            _ColopolaAugmenter(
                rotation_prob=rotation_prob,
                rotation_deg=rotation_deg,
                flip_prob=flip_prob,
            )
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _to_16ch(mu_np: np.ndarray) -> torch.Tensor:
        if mu_np.ndim == 5:
            mu_np = mu_np[0]
        if mu_np.ndim == 4 and mu_np.shape[:2] == (4, 4):
            h, w = mu_np.shape[2], mu_np.shape[3]
            mu_np = mu_np.reshape(16, h, w)
        elif mu_np.ndim == 3 and mu_np.shape[0] == 16:
            pass
        else:
            raise ValueError(f"Unexpected Mueller array shape: {mu_np.shape}")
        mu_np = np.nan_to_num(mu_np, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(mu_np.astype(np.float32))

    @staticmethod
    def _center_crop_or_pad(x: torch.Tensor, crop_size: int) -> torch.Tensor:
        _, h, w = x.shape
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
            _, h, w = x.shape

        top = max(0, (h - crop_size) // 2)
        left = max(0, (w - crop_size) // 2)
        return x[:, top : top + crop_size, left : left + crop_size]

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
        _, h, w = x.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

    @staticmethod
    def _enforce_unit_range(x: torch.Tensor) -> torch.Tensor:
        # Keep all Mueller inputs in [-1, 1] via direct clipping.
        return torch.clamp(x, min=-1.0, max=1.0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        mu_np = np.load(sample.path, allow_pickle=False)
        x = self._to_16ch(mu_np)

        if self.augmenter is not None:
            x = self.augmenter(x)

        if self.center_crop_size is not None:
            x = self._center_crop_or_pad(x, self.center_crop_size)
        if self.pad_multiple is not None:
            x = self._pad_to_multiple(x, self.pad_multiple)
        x = self._enforce_unit_range(x)

        y = torch.tensor(sample.label, dtype=torch.long)
        return x, y


def summarize_class_counts(samples: Iterable[ColopolaSample]) -> Dict[int, int]:
    counts = {0: 0, 1: 0}
    for sample in samples:
        counts[sample.label] += 1
    return counts
