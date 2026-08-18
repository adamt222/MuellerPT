from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from augmentations.mueller_rotation import RandomMuellerRotationExpandCrop


class FilteredMuellerDataset(Dataset):
    """Load the fixed 16-channel MuellerPT pretraining inputs and M00 maps."""

    def __init__(
        self,
        dataset_name: str,
        filtered_root: Path,
        paths: Iterable[Path],
        center_crop_size: tuple[int, int],
        transform: RandomMuellerRotationExpandCrop,
        m00_root: Path,
        m00_suffix: str,
    ) -> None:
        self.dataset_name = dataset_name
        self.filtered_root = Path(filtered_root)
        self.paths = [Path(path) for path in paths]
        self.center_crop_size = center_crop_size
        self.transform = transform
        self.m00_root = Path(m00_root)
        self.m00_suffix = m00_suffix

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _to_16ch(array: np.ndarray) -> torch.Tensor:
        if array.ndim == 5:
            array = array[0]
        if array.ndim == 4 and array.shape[:2] == (4, 4):
            array = array.reshape(16, array.shape[2], array.shape[3])
        elif array.ndim != 3 or array.shape[0] != 16:
            raise ValueError(f"Unexpected Mueller shape: {array.shape}")
        return torch.from_numpy(array.astype(np.float32))

    def _placeholder(self) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = self.center_crop_size
        return (
            torch.zeros((16, height, width), dtype=torch.float32),
            torch.zeros((1, height, width), dtype=torch.float32),
        )

    def _m00_path(self, filtered_path: Path) -> Path:
        relative = filtered_path.relative_to(self.filtered_root)
        return self.m00_root / relative.parent / f"{filtered_path.stem}{self.m00_suffix}"

    def _load_m00(
        self, filtered_path: Path, height: int, width: int
    ) -> tuple[torch.Tensor, bool]:
        try:
            array = np.load(self._m00_path(filtered_path), allow_pickle=False)
            if array.size == 0:
                raise ValueError("empty M00 array")
            if array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 2:
                raise ValueError(f"Unexpected M00 shape: {array.shape}")
            m00 = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
            if m00.shape[-2:] != (height, width):
                m00 = F.interpolate(
                    m00.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            return m00, True
        except Exception:
            return torch.zeros((1, height, width), dtype=torch.float32), False

    def _metadata(self, path: Path, *, skip: bool) -> dict[str, object]:
        return {
            "skip": skip,
            "m00_present": False,
            "rotation_applied": False,
            "rotation_angle": 0.0,
            "rotation_expand": True,
            "rotation_center": (float("nan"), float("nan")),
            "center_crop": self.center_crop_size,
            "dataset": self.dataset_name,
            "path": str(path),
        }

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            array = np.load(path, allow_pickle=False)
            if array.size == 0:
                raise ValueError("empty Mueller array")
            x = self._to_16ch(array)
        except Exception:
            x, m00 = self._placeholder()
            return x, m00, self._metadata(path, skip=True)

        m00, m00_present = self._load_m00(path, x.shape[1], x.shape[2])
        crop_height, crop_width = self.center_crop_size
        if x.shape[-2:] != self.center_crop_size:
            x = TF.center_crop(x, [crop_height, crop_width])
            m00 = TF.center_crop(m00, [crop_height, crop_width])

        metadata = self._metadata(path, skip=False)
        if random.random() < self.transform.p:
            angle = self.transform.get_params(self.transform.degrees)
            x = self.transform.apply_rotation(x, angle=angle)
            m00 = TF.rotate(
                m00,
                angle,
                interpolation=TF.InterpolationMode.BILINEAR,
                expand=True,
                center=self.transform.center,
                fill=0,
            )
            m00 = TF.center_crop(m00, [crop_height, crop_width])
            metadata["rotation_applied"] = True
            metadata["rotation_angle"] = float(angle)

        metadata["m00_present"] = bool(m00_present)
        return x, m00, metadata
