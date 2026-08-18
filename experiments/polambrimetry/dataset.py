from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(__file__).resolve().parent
PRETRAINING_ROOT = REPO_ROOT / "pretraining"
for path in (LOCAL, PRETRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import GM_LABELS, WM_LABELS, IGNORE_INDEX
from augmentations.mueller_rotation import RandomMuellerRotationCustom

DEFAULT_INPUT_ROOT = Path(os.environ.get("MUELLERPT_INPUT_ROOT", str(REPO_ROOT / "data")))
DEFAULT_POLAMBRIMETRY_ROOT = Path(os.environ.get("MUELLERPT_POLAMBRIMETRY_ROOT", str(DEFAULT_INPUT_ROOT / "polambrimetry")))


def _map_labels(labels: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    if num_classes == 2:
        mapped = np.full(labels.shape, ignore_index, dtype=np.int64)
    else:
        mapped = np.zeros(labels.shape, dtype=np.int64)
    gm_mask = np.isin(labels, list(GM_LABELS))
    wm_mask = np.isin(labels, list(WM_LABELS))
    if num_classes == 2:
        mapped[gm_mask] = 0
        mapped[wm_mask] = 1
    else:
        mapped[gm_mask] = 1
        mapped[wm_mask] = 2
    return mapped


@dataclass
class SampleIndex:
    specimen_id: str
    path: Path
    wavelength: int


class PolambrimetryFullDataset(Dataset):
    def __init__(
        self,
        specimen_ids: List[str],
        augment: bool = False,
        rotation_prob: float = 0.8,
        flip_prob: float = 0.8,
        rotation_deg: float = 45.0,
        patch_size: int | None = None,
        patch_max_bg_frac: float = 0.5,
        patch_max_tries: int = 10,
        patch_balance_gm_wm: bool = False,
        center_crop_size: int | None = None,
        num_classes: int = 3,
        ignore_index: int = IGNORE_INDEX,
        use_m00: bool = False,
        m00_log_input: bool = False,
        m00_log_eps: float = 1e-6,
    ):
        self.specimen_ids = specimen_ids
        self.augment = augment
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.bg_label = ignore_index if num_classes == 2 else 0
        self.gm_label = 0 if num_classes == 2 else 1
        self.wm_label = 1 if num_classes == 2 else 2
        self.samples: List[SampleIndex] = []

        print(f"[DATASET] Building dataset with {len(specimen_ids)} specimen IDs (augment={augment})")
        file_count = 0
        m00_file_count = 0

        for specimen_id in specimen_ids:
            for path in _resolve_h5_paths(specimen_id):
                file_count += 1
                with h5py.File(path, "r") as f:
                    if "m00" in f:
                        m00_file_count += 1
                    m = f["M"]
                    wavelengths = m.shape[0]
                for w in range(wavelengths):
                    self.samples.append(
                        SampleIndex(specimen_id=specimen_id, path=path, wavelength=w)
                    )

        print(
            f"[DATASET] Total h5 files: {file_count}, m00 files found: {m00_file_count}, "
            f"total samples: {len(self.samples)}"
        )

        self._augmenter = None
        if augment:
            self._augmenter = _SSLRotationAugmenter(
                rotation_prob=rotation_prob,
                rotation_deg=rotation_deg,
                fill=self.bg_label,
            )
        self.patch_size = patch_size
        self.patch_max_bg_frac = patch_max_bg_frac
        self.patch_max_tries = patch_max_tries
        self.patch_balance_gm_wm = patch_balance_gm_wm
        self.center_crop_size = center_crop_size
        self.use_m00 = use_m00
        self.m00_log_input = m00_log_input
        self.m00_log_eps = m00_log_eps
        self._missing_m00_warned_paths: set[Path] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        with h5py.File(sample.path, "r") as f:
            mueller = f["M"][sample.wavelength]  # (4,4,H,W)
            labels = f["labels"][()]
            if self.use_m00:
                if "m00" in f:
                    m00_data = f["m00"][()]
                    if m00_data.ndim == 3:
                        m00 = m00_data[sample.wavelength]
                    elif m00_data.ndim == 2:
                        m00 = m00_data
                    elif m00_data.ndim == 1:
                        raise ValueError(f"Unexpected m00 shape in {sample.path}: {m00_data.shape}")
                    else:
                        raise ValueError(f"Unexpected m00 shape in {sample.path}: {m00_data.shape}")
                else:
                    if sample.path not in self._missing_m00_warned_paths:
                        print(f"[DATASET] WARNING: no 'm00' found in {sample.path}; using zeros.")
                        self._missing_m00_warned_paths.add(sample.path)
                    m00 = np.zeros_like(mueller[0, 0], dtype=np.float32)
            else:
                m00 = np.zeros_like(mueller[0, 0], dtype=np.float32)

        mueller = np.nan_to_num(mueller, nan=0.0, posinf=0.0, neginf=0.0)
        m00 = np.nan_to_num(m00, nan=0.0, posinf=0.0, neginf=0.0)
        labels = labels.astype(np.int64)
        labels = _map_labels(labels, self.num_classes, self.ignore_index)

        mueller_t = torch.from_numpy(mueller.astype(np.float32)).view(16, mueller.shape[2], mueller.shape[3])
        m00_t = torch.from_numpy(m00.astype(np.float32)).unsqueeze(0)
        if self.m00_log_input:
            m00_t = torch.log(m00_t.clamp_min(self.m00_log_eps))
        label_t = torch.from_numpy(labels)

        if self._augmenter is not None:
            mueller_t, label_t, m00_t = self._augmenter(mueller_t, label_t, m00_t)

        if self.center_crop_size is not None:
            mueller_t, label_t = _center_crop(
                mueller_t,
                label_t,
                crop_size=self.center_crop_size,
                pad_value=self.bg_label,
            )
            m00_t = TF.center_crop(m00_t, [self.center_crop_size, self.center_crop_size])

        if self.patch_size is not None:
            mueller_t, label_t, m00_t = _sample_patch(
                mueller_t,
                label_t,
                patch_size=self.patch_size,
                max_bg_frac=self.patch_max_bg_frac,
                max_tries=self.patch_max_tries,
                balance_gm_wm=self.patch_balance_gm_wm,
                bg_label=self.bg_label,
                gm_label=self.gm_label,
                wm_label=self.wm_label,
                aux_image=m00_t,
            )

        mueller_t, label_t, m00_t = _pad_to_multiple(
            mueller_t, label_t, multiple=16, pad_value=self.bg_label, aux_image=m00_t
        )
        if m00_t is None:
            m00_t = torch.zeros((1, mueller_t.shape[1], mueller_t.shape[2]), dtype=mueller_t.dtype)
        return mueller_t, m00_t, label_t


def _pad_to_multiple(
    image: torch.Tensor,
    label: torch.Tensor,
    multiple: int,
    pad_value: int,
    aux_image: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    _, h, w = image.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, label, aux_image
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
    label_padded = F.pad(label, (0, pad_w, 0, pad_h), value=pad_value)
    if aux_image is None:
        return image_padded, label_padded, None
    aux_padded = F.pad(aux_image, (0, pad_w, 0, pad_h), value=0.0)
    return image_padded, label_padded, aux_padded


def _center_crop(
    image: torch.Tensor,
    label: torch.Tensor,
    crop_size: int,
    pad_value: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, h, w = image.shape
    crop_h = crop_size
    crop_w = crop_size
    pad_h = max(0, crop_h - h)
    pad_w = max(0, crop_w - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        label = F.pad(label, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
        _, h, w = image.shape
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    image = image[:, top : top + crop_h, left : left + crop_w]
    label = label[top : top + crop_h, left : left + crop_w]
    return image, label


def _resolve_h5_paths(specimen_id: str) -> List[Path]:
    candidate = Path(specimen_id)
    if candidate.is_file() and candidate.suffix.lower() in {".h5", ".hdf5"}:
        return [candidate]

    specimen_dir = candidate
    if not specimen_dir.is_dir():
        specimen_dir = DEFAULT_POLAMBRIMETRY_ROOT / specimen_id
    if not specimen_dir.is_dir():
        raise FileNotFoundError(specimen_id)

    paths = sorted(specimen_dir.rglob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No .h5 files under {specimen_dir}")
    return paths


class _SSLRotationAugmenter:
    def __init__(
        self,
        rotation_prob: float = 0.8,
        rotation_deg: float = 45.0,
        fill: int = 0,
    ):
        self.rotate = RandomMuellerRotationCustom(
            degrees=rotation_deg,
            p=rotation_prob,
            any=True,
            fill=fill,
        )

    def __call__(
        self,
        frame: torch.Tensor,
        label: torch.Tensor,
        m00: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        label_in = label
        if label.dim() == 2:
            label = label.unsqueeze(0)
        label = label.float()

        if random.random() < self.rotate.p:
            angle = self.rotate.get_params(self.rotate.degrees)
            frame, label = self.rotate(frame, label=label, angle=angle)
            if m00 is not None:
                m00 = TF.rotate(
                    m00,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    expand=self.rotate.expand,
                    center=self.rotate.center,
                    fill=0,
                )

        if label is None:
            return frame, label_in, m00
        if label.dim() == 3:
            label = label.squeeze(0)
        label = label.round().to(dtype=label_in.dtype)
        return frame, label, m00


def _sample_patch(
    image: torch.Tensor,
    label: torch.Tensor,
    patch_size: int,
    max_bg_frac: float,
    max_tries: int,
    balance_gm_wm: bool = False,
    bg_label: int = 0,
    gm_label: int = 1,
    wm_label: int = 2,
    aux_image: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    _, h, w = image.shape
    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
        label = F.pad(label, (0, pad_w, 0, pad_h), value=bg_label)
        if aux_image is not None:
            aux_image = F.pad(aux_image, (0, pad_w, 0, pad_h), value=0.0)
        _, h, w = image.shape

    best_top = 0
    best_left = 0
    best_bg = 1.0
    target_majority = None
    best_majority_score = float("-inf")
    best_majority_top = 0
    best_majority_left = 0

    if balance_gm_wm:
        target_majority = 1 if torch.rand(()) < 0.5 else 2
    for _ in range(max_tries):
        top = int(torch.randint(0, h - patch_size + 1, (1,)).item())
        left = int(torch.randint(0, w - patch_size + 1, (1,)).item())
        label_patch = label[top : top + patch_size, left : left + patch_size]
        bg_frac = (label_patch == bg_label).float().mean().item()
        if bg_frac <= max_bg_frac:
            if not balance_gm_wm:
                return (
                    image[:, top : top + patch_size, left : left + patch_size],
                    label_patch,
                    None
                    if aux_image is None
                    else aux_image[:, top : top + patch_size, left : left + patch_size],
                )
            gm_count = int((label_patch == gm_label).sum().item())
            wm_count = int((label_patch == wm_label).sum().item())
            if target_majority == 1:
                if gm_count > wm_count and gm_count > 0:
                    return (
                        image[:, top : top + patch_size, left : left + patch_size],
                        label_patch,
                        None
                        if aux_image is None
                        else aux_image[:, top : top + patch_size, left : left + patch_size],
                    )
                score = gm_count - wm_count
            else:
                if wm_count > gm_count and wm_count > 0:
                    return (
                        image[:, top : top + patch_size, left : left + patch_size],
                        label_patch,
                        None
                        if aux_image is None
                        else aux_image[:, top : top + patch_size, left : left + patch_size],
                    )
                score = wm_count - gm_count
            if score > best_majority_score:
                best_majority_score = score
                best_majority_top = top
                best_majority_left = left
        if bg_frac < best_bg:
            best_bg = bg_frac
            best_top = top
            best_left = left

    if balance_gm_wm and best_majority_score > float("-inf"):
        return (
            image[
                :,
                best_majority_top : best_majority_top + patch_size,
                best_majority_left : best_majority_left + patch_size,
            ],
            label[
                best_majority_top : best_majority_top + patch_size,
                best_majority_left : best_majority_left + patch_size,
            ],
            None
            if aux_image is None
            else aux_image[
                :,
                best_majority_top : best_majority_top + patch_size,
                best_majority_left : best_majority_left + patch_size,
            ],
        )
    return (
        image[:, best_top : best_top + patch_size, best_left : best_left + patch_size],
        label[best_top : best_top + patch_size, best_left : best_left + patch_size],
        None
        if aux_image is None
        else aux_image[:, best_top : best_top + patch_size, best_left : best_left + patch_size],
    )


def build_folds(specimen_ids: List[str], folds: int) -> List[Tuple[List[str], List[str]]]:
    if folds <= 1:
        raise ValueError("folds must be >= 2")
    ids = list(specimen_ids)
    fold_sets: List[Tuple[List[str], List[str]]] = []
    for i in range(folds):
        test = [ids[i % len(ids)]]
        train = [x for x in ids if x not in test]
        fold_sets.append((train, test))
    return fold_sets


def load_polambrimetry_ids(max_specimens: int | None = None, root: Path = DEFAULT_POLAMBRIMETRY_ROOT) -> List[str]:
    specimen_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("specimen_")]
    )
    ids = [str(p) for p in specimen_dirs]
    if max_specimens is not None:
        ids = ids[:max_specimens]
    return ids
