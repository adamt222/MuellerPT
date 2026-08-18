from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

PRETRAINING_ROOT = Path(__file__).resolve().parent
if str(PRETRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAINING_ROOT))

from config_pretrain import (
    BATCH_SIZE,
    CHANNEL_DROP_KEEP_FULL_PROB,
    CHANNEL_DROP_PRESETS,
    DECOMP_CACHE_SUFFIX,
    DECOMP_ORDER,
    DEVICE,
    EPOCHS,
    FILTERED_MATRICES_ROOT,
    GRAD_CLIP_MAX_NORM,
    LR,
    LR_PLATEAU_FACTOR,
    LR_PLATEAU_MIN_LR,
    LR_PLATEAU_PATIENCE,
    M00_DROP_PROB,
    NUM_WORKERS,
    PIN_MEMORY,
    SAVE_EVERY_N_EPOCHS,
    SEED,
    WEIGHT_DECAY,
)
from models.encoders import HRNetEncoder
from models.heads import HRNetDecompHead
from models.pretrain import MuellerPretrainModel
from pretrain_data import build_pretrain_dataset
from pretrain_run import append_metrics, init_run_dir, write_run_config


def _batch_value(metadata: dict, key: str, index: int, default):
    if key not in metadata:
        return default
    values = metadata[key]
    if isinstance(values, (list, tuple)):
        if not values:
            return default
        if torch.is_tensor(values[0]) and values[0].ndim >= 1:
            return [value[index].item() for value in values]
        return values[index] if index < len(values) else default
    if torch.is_tensor(values):
        if values.ndim == 0:
            return values.item()
        return values[index] if index < values.shape[0] else default
    return values


def _filter_skipped_batch(
    x: torch.Tensor,
    m00: torch.Tensor,
    metadata: dict,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict]:
    skipped = metadata.get("skip")
    if skipped is None:
        return x, m00, metadata
    if torch.is_tensor(skipped):
        keep = ~skipped.to(dtype=torch.bool)
    else:
        keep = ~torch.tensor(skipped, dtype=torch.bool)
    if not keep.any():
        return None, None, metadata

    filtered: dict[str, object] = {}
    for key, value in metadata.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == keep.shape[0]:
            filtered[key] = value[keep]
        elif isinstance(value, (list, tuple)):
            # Default collation transposes tuple metadata such as crop sizes.
            if value and torch.is_tensor(value[0]) and value[0].shape[0] == keep.shape[0]:
                filtered[key] = [item[keep] for item in value]
            elif len(value) == keep.shape[0]:
                filtered[key] = [item for item, include in zip(value, keep) if include]
            else:
                filtered[key] = value
        else:
            filtered[key] = value
    return x[keep], m00[keep], filtered


def _rotation_parameters(
    metadata: dict,
    index: int,
) -> tuple[bool, float, tuple[float, float] | None]:
    applied = bool(_batch_value(metadata, "rotation_applied", index, False))
    angle = float(_batch_value(metadata, "rotation_angle", index, 0.0))
    center_value = _batch_value(metadata, "rotation_center", index, None)
    center = tuple(center_value) if isinstance(center_value, (list, tuple)) else None
    if center is not None and any(math.isnan(float(value)) for value in center):
        center = None
    return applied, angle, center


def _transform_cached_map(
    array: np.ndarray,
    metadata: dict,
    index: int,
    interpolation: TF.InterpolationMode,
) -> np.ndarray:
    crop = _batch_value(metadata, "center_crop", index, None)
    if isinstance(crop, (list, tuple)) and len(crop) >= 2:
        tensor = TF.center_crop(
            torch.from_numpy(array).unsqueeze(0),
            [int(crop[0]), int(crop[1])],
        )
        array = tensor.squeeze(0).numpy()

    applied, angle, center = _rotation_parameters(metadata, index)
    if applied:
        height, width = array.shape
        tensor = TF.rotate(
            torch.from_numpy(array).unsqueeze(0),
            angle,
            interpolation=interpolation,
            expand=True,
            center=center,
            fill=0,
        )
        array = TF.center_crop(tensor, [height, width]).squeeze(0).numpy()
    return array


def _decomp_cache_path(filtered_path: str) -> Path:
    path = Path(filtered_path)
    return path.with_name(f"{path.stem}{DECOMP_CACHE_SUFFIX}")


def _load_cached_targets(metadata: dict) -> dict[str, torch.Tensor]:
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in DECOMP_ORDER}
    for index, path in enumerate(metadata["path"]):
        cache_path = _decomp_cache_path(path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing decomposition cache: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cache:
            for name in DECOMP_ORDER:
                array = _transform_cached_map(
                    cache[name],
                    metadata,
                    index,
                    TF.InterpolationMode.BILINEAR,
                )
                upper = np.pi if name == "retardance" else 1.0
                array = np.clip(array, 0.0, upper)
                targets[name].append(torch.from_numpy(array).unsqueeze(0))
    return {
        name: torch.stack(items).float().to(DEVICE, non_blocking=True)
        for name, items in targets.items()
    }


def _rotation_valid_mask(
    metadata: dict,
    batch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    masks = []
    for index in range(batch_size):
        mask = torch.ones((1, height, width), dtype=torch.float32)
        applied, angle, center = _rotation_parameters(metadata, index)
        if applied:
            mask = TF.rotate(
                mask,
                angle,
                interpolation=TF.InterpolationMode.NEAREST,
                expand=True,
                center=center,
                fill=0,
            )
            mask = TF.center_crop(mask, [height, width])
        masks.append(mask)
    return torch.stack(masks).to(DEVICE, non_blocking=True)


def _prepare_batch(
    x: torch.Tensor,
    m00: torch.Tensor,
    metadata: dict,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    targets = _load_cached_targets(metadata)
    x = x.to(DEVICE, non_blocking=True)
    m00 = m00.to(DEVICE, non_blocking=True).float()
    if m00.ndim != 4 or m00.shape[1] != 1:
        raise ValueError(f"Expected M00 shape [B,1,H,W], got {tuple(m00.shape)}")
    present = metadata["m00_present"].to(
        device=DEVICE, dtype=torch.bool, non_blocking=True
    )
    m00 = m00 * present.view(-1, 1, 1, 1).to(m00.dtype)
    valid_mask = _rotation_valid_mask(
        metadata, x.shape[0], x.shape[2], x.shape[3]
    )
    return x, m00, present, targets, valid_mask


def _preset_drop_indices(name: str) -> list[int]:
    if name == "upper_left_3x3":
        retained = {row * 4 + column for row in range(3) for column in range(3)}
        return [index for index in range(16) if index not in retained]
    if name == "drop_last_column":
        return [3, 7, 11, 15]
    if name == "drop_last_row":
        return [12, 13, 14, 15]
    raise ValueError(f"Unknown channel-drop preset: {name}")


def _channel_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch_size, 16), dtype=torch.bool, device=device)
    keep_full = torch.rand((batch_size,), device=device) < CHANNEL_DROP_KEEP_FULL_PROB
    choices = torch.randint(
        0, len(CHANNEL_DROP_PRESETS), (batch_size,), device=device
    )
    for index in range(batch_size):
        if bool(keep_full[index].item()):
            continue
        preset = CHANNEL_DROP_PRESETS[int(choices[index].item())]
        mask[index, _preset_drop_indices(preset)] = True
    return mask


def train() -> None:
    torch.manual_seed(SEED)
    run_id, run_dir = init_run_dir()
    write_run_config(run_dir)

    dataset = build_pretrain_dataset()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

    encoder = HRNetEncoder()
    decoder = HRNetDecompHead(
        in_channels=encoder.highres_channels,
        decomp_order=DECOMP_ORDER,
    )
    model = MuellerPretrainModel(
        encoder=encoder,
        decomp_decoder=decoder,
        decomp_order=DECOMP_ORDER,
        m00_drop_prob=M00_DROP_PROB,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_PLATEAU_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
        min_lr=LR_PLATEAU_MIN_LR,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for step, (x, m00, metadata) in enumerate(loader, start=1):
            x, m00, metadata = _filter_skipped_batch(x, m00, metadata)
            if x is None or m00 is None:
                continue
            x, m00, present, targets, valid_mask = _prepare_batch(x, m00, metadata)
            channel_mask = _channel_mask(x.shape[0], x.device)

            optimizer.zero_grad(set_to_none=True)
            output = model(
                x,
                targets=targets,
                valid_mask=valid_mask,
                m00=m00,
                m00_present=present,
                channel_mask=channel_mask,
            )
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP_MAX_NORM
            )
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            if step % 20 == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"loss={total_loss / max(1, steps):.4f}"
                )

        average_loss = total_loss / max(1, steps)
        scheduler.step(average_loss)
        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                run_dir / f"epoch_{epoch:03d}.pt",
            )
        append_metrics(run_dir, run_id, epoch, average_loss)


if __name__ == "__main__":
    train()
