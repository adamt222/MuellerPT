from __future__ import annotations

from pathlib import Path

from augmentations.mueller_rotation import RandomMuellerRotationExpandCrop
from config_pretrain import (
    CENTER_CROP_SIZE,
    DATASET_NAME,
    FILTERED_MATRICES_ROOT,
    M00_ROOT,
    M00_SUFFIX,
    ROTATION_DEGREES,
    ROTATION_P,
)
from data import FilteredMuellerDataset


def build_pretrain_dataset() -> FilteredMuellerDataset:
    dataset_dir = FILTERED_MATRICES_ROOT / DATASET_NAME
    paths = sorted(dataset_dir.rglob("*_filtered.npy"))
    if not paths:
        raise FileNotFoundError(
            f"No *_filtered.npy inputs found beneath {dataset_dir}."
        )
    print(f"[pretrain] {DATASET_NAME} files={len(paths)}")
    rotation = RandomMuellerRotationExpandCrop(
        degrees=ROTATION_DEGREES,
        p=ROTATION_P,
        any=True,
        expand=True,
    )
    return FilteredMuellerDataset(
        dataset_name=DATASET_NAME,
        filtered_root=FILTERED_MATRICES_ROOT,
        paths=paths,
        center_crop_size=CENTER_CROP_SIZE,
        transform=rotation,
        m00_root=Path(M00_ROOT),
        m00_suffix=M00_SUFFIX,
    )
