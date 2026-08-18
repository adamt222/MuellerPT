from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from datetime import datetime
import math
from typing import List, Tuple, Optional, Dict, Iterable
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import TrainConfig, resolve_device, IGNORE_INDEX
from dataset import (
    PolambrimetryFullDataset,
    build_folds,
    load_polambrimetry_ids,
)
from model import HRNetSegmentation


def _prepare_m00_tensor(
    m00: Optional[torch.Tensor], target_hw: tuple[int, int]
) -> torch.Tensor:
    if m00 is None:
        raise ValueError("m00 tensor is required when m00 fusion is enabled.")
    if m00.ndim == 3:
        m00 = m00.unsqueeze(1)
    if m00.ndim != 4 or m00.shape[1] != 1:
        raise ValueError(f"Expected m00 shape [B,1,H,W], got {tuple(m00.shape)}")
    if m00.shape[-2:] != target_hw:
        m00 = torch.nn.functional.interpolate(
            m00, size=target_hw, mode="bilinear", align_corners=False
        )
    return m00


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_run_config(
    outdir: Path,
    cfg: TrainConfig,
    args: argparse.Namespace,
    overrides: Optional[Dict[str, object]] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cfg_dict = _json_safe(asdict(cfg))
    args_dict = _json_safe(vars(args))
    payload = {
        "config": cfg_dict,
        "args": args_dict,
    }
    if overrides:
        payload["overrides"] = _json_safe(overrides)
    (outdir / "config_used.json").write_text(json.dumps(payload, indent=2))
    # Keep a snapshot of config.py for reference
    src = Path(__file__).resolve().parent / "config.py"
    dst = outdir / "config_used.py"
    if src.exists():
        dst.write_text(src.read_text())


def _make_run_id(prefix: Optional[str]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}" if prefix else stamp


def _as_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_few_shot_percentages(values: object) -> List[int]:
    if values is None:
        return []
    if isinstance(values, (int, float)):
        values = [values]
    if not isinstance(values, (list, tuple)):
        raise ValueError("few_shot_percentages must be a list of percentages.")
    out: List[int] = []
    for value in values:
        pct = int(value)
        if pct <= 0 or pct > 100:
            raise ValueError(f"few_shot percentage must be in [1, 100], got {pct}")
        out.append(pct)
    return out


def _build_config(args: argparse.Namespace) -> TrainConfig:
    def pick(name: str, default):
        val = getattr(args, name, None)
        if val is None:
            return default
        return val

    dataset_root = pick("dataset_root", TrainConfig.dataset_root)
    if isinstance(dataset_root, str):
        dataset_root = Path(dataset_root)

    seg_ckpt = pick("seg_encoder_checkpoint", TrainConfig.seg_encoder_checkpoint)
    default_few_shot_percentages = list(TrainConfig().few_shot_percentages)
    few_shot_percentages = _normalize_few_shot_percentages(
        pick("few_shot_percentages", default_few_shot_percentages)
    )

    return TrainConfig(
        dataset_root=dataset_root,
        batch_size=pick("batch_size", TrainConfig.batch_size),
        num_workers=pick("num_workers", TrainConfig.num_workers),
        epochs=pick("epochs", TrainConfig.epochs),
        lr=pick("lr", TrainConfig.lr),
        weight_decay=pick("weight_decay", TrainConfig.weight_decay),
        device=resolve_device(pick("device", TrainConfig.device)),
        folds=pick("folds", TrainConfig.folds),
        max_specimens=pick("max_specimens", TrainConfig.max_specimens),
        few_shot_percentages=few_shot_percentages,
        rotation_deg=pick("rotation_deg", TrainConfig.rotation_deg),
        rotation_prob=pick("rotation_prob", TrainConfig.rotation_prob),
        flip_prob=pick("flip_prob", TrainConfig.flip_prob),
        num_classes=pick("num_classes", TrainConfig.num_classes),
        encoder_variant=pick("encoder_variant", TrainConfig.encoder_variant),
        seg_encoder_checkpoint=_as_str(seg_ckpt),
        center_crop_size=pick("center_crop_size", TrainConfig.center_crop_size),
        lr_plateau_factor=pick("lr_plateau_factor", TrainConfig.lr_plateau_factor),
        lr_plateau_patience=pick("lr_plateau_patience", TrainConfig.lr_plateau_patience),
        lr_plateau_min_lr=pick("lr_plateau_min_lr", TrainConfig.lr_plateau_min_lr),
        nested_early_stop_patience=pick(
            "nested_early_stop_patience", TrainConfig.nested_early_stop_patience
        ),
        early_stop_min_delta=pick("early_stop_min_delta", TrainConfig.early_stop_min_delta),
        seed=pick("seed", TrainConfig.seed),
        deterministic=pick("deterministic", TrainConfig.deterministic),
        m00_log_eps=pick("m00_log_eps", TrainConfig.m00_log_eps),
    )


def _run_training(
    cfg: TrainConfig,
    args: argparse.Namespace,
    outdir_root: Path,
    run_id: str,
    fold_index: Optional[int],
) -> None:
    ids = load_polambrimetry_ids(cfg.max_specimens, cfg.dataset_root)
    folds = build_folds(ids, cfg.folds)

    outdir = outdir_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    _write_run_config(outdir, cfg, args)
    _set_determinism(cfg.seed, cfg.deterministic)
    print(f"[RUN] seed={cfg.seed} deterministic={cfg.deterministic}")
    seg_encoder_checkpoint = (
        Path(cfg.seg_encoder_checkpoint) if cfg.seg_encoder_checkpoint else None
    )
    if seg_encoder_checkpoint is None:
        raise ValueError("MuellerPT checkpoint is required for the paper comparison.")
    if not seg_encoder_checkpoint.is_file():
        raise FileNotFoundError(
            "MuellerPT checkpoint not found: "
            f"{seg_encoder_checkpoint}. Set MUELLERPT_CHECKPOINT or pass "
            "--seg-encoder-checkpoint."
        )
    if not cfg.few_shot_percentages:
        raise ValueError("At least one few-shot percentage is required.")
    few_shot_percentages = [int(pct) for pct in cfg.few_shot_percentages]
    print(f"[RUN] paper few-shot percentages={few_shot_percentages}")

    resume_enabled = bool(getattr(args, "resume", False))
    fold_label = "pair_index"
    summary_path = outdir / "nested_summary.csv"
    completed_keys = (
        _load_completed_run_keys(summary_path, nested=True)
        if resume_enabled
        else set()
    )
    if resume_enabled:
        print(f"[RESUME] loaded {len(completed_keys)} completed entries from {summary_path}")

    def _run_key(run_name: str, fold_idx: int, few_shot_pct: int | None) -> tuple[str, int, int]:
        return run_name, fold_idx, _few_shot_to_key(few_shot_pct)

    def _is_completed(run_name: str, fold_idx: int, few_shot_pct: int | None) -> bool:
        if not resume_enabled:
            return False
        key = _run_key(run_name, fold_idx, few_shot_pct)
        if key in completed_keys:
            pct = _few_shot_to_key(few_shot_pct)
            print(f"[RESUME] skipping {run_name} {fold_label}={fold_idx} few_shot={pct}")
            return True
        return False

    def _run_pair(
        fold_idx: int,
        train_ids_split: List[str],
        val_ids: List[str],
        test_ids: List[str],
        nested_info: Optional[Dict[str, object]] = None,
    ) -> None:
        for few_shot_pct in few_shot_percentages:
            baseline: Tuple[int, float, List[float], float] | None = None
            if not _is_completed("ssl_pretrained", fold_idx, few_shot_pct):
                baseline = _run_fold(
                    fold_idx,
                    train_ids_split,
                    val_ids,
                    test_ids,
                    cfg,
                    outdir,
                    cfg.device,
                    "ssl_pretrained",
                    seg_encoder_checkpoint,
                    nested_info=nested_info,
                    few_shot_pct=few_shot_pct,
                )
                completed_keys.add(_run_key("ssl_pretrained", fold_idx, few_shot_pct))
            if not _is_completed("random", fold_idx, few_shot_pct):
                _run_fold(
                    fold_idx,
                    train_ids_split,
                    val_ids,
                    test_ids,
                    cfg,
                    outdir,
                    cfg.device,
                    "random",
                    None,
                    baseline=baseline,
                    nested_info=nested_info,
                    few_shot_pct=few_shot_pct,
                )
                completed_keys.add(_run_key("random", fold_idx, few_shot_pct))

    pair_index = 0
    for outer_fold_idx, (train_ids, test_ids) in enumerate(folds):
        for val_id in train_ids:
            if fold_index is None or outer_fold_idx == fold_index:
                train_ids_split = [x for x in train_ids if x != val_id]
                val_ids = [val_id]
                nested_info = {
                    "pair_index": pair_index,
                    "outer_fold": outer_fold_idx,
                    "train_ids": train_ids_split,
                    "val_ids": val_ids,
                    "test_ids": test_ids,
                }
                _run_pair(
                    pair_index,
                    train_ids_split,
                    val_ids,
                    test_ids,
                    nested_info=nested_info,
                )
            pair_index += 1


def _set_determinism(seed: int, enable: bool) -> None:
    if not enable:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch: no warn_only flag
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            print(f"[WARN] Deterministic algorithms not fully enabled: {exc}")


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
):
    model.train()
    total_loss = 0.0
    data_time = 0.0
    step_time = 0.0
    end = time.perf_counter()
    for inputs, m00, targets in loader:
        data_time += time.perf_counter() - end
        step_start = time.perf_counter()
        # print(f"Input batch shape: {inputs.shape}, Target batch shape: {targets.shape}")
        inputs = inputs.to(device)
        m00 = m00.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        m00_ready = _prepare_m00_tensor(m00, target_hw=inputs.shape[-2:])
        m00_present = torch.ones((inputs.shape[0],), dtype=torch.bool, device=inputs.device)
        logits = model(inputs, m00=m00_ready, m00_present=m00_present)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        step_time += time.perf_counter() - step_start
        end = time.perf_counter()
    return total_loss / max(1, len(loader)), data_time, step_time


def _compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    ious = torch.zeros(num_classes, dtype=torch.float64)
    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    for cls in range(num_classes):
        pred_mask = (preds == cls) & mask
        target_mask = (targets == cls) & mask
        inter = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            ious[cls] = float("nan")
        else:
            ious[cls] = inter / union
    return ious


def _compute_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    dices = torch.zeros(num_classes, dtype=torch.float64)
    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    for cls in range(num_classes):
        pred_mask = (preds == cls) & mask
        target_mask = (targets == cls) & mask
        inter = (pred_mask & target_mask).sum().item()
        denom = pred_mask.sum().item() + target_mask.sum().item()
        if denom == 0:
            dices[cls] = float("nan")
        else:
            dices[cls] = 2.0 * inter / denom
    return dices


def _dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if ignore_index is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        mask = targets != ignore_index
    if mask.sum().item() == 0:
        return logits.sum() * 0.0

    probs = torch.softmax(logits, dim=1)
    targets_safe = targets
    if ignore_index is not None:
        targets_safe = targets.clone()
        targets_safe[targets == ignore_index] = 0
    one_hot = torch.nn.functional.one_hot(targets_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        probs = probs[:, 1:, ...]
        one_hot = one_hot[:, 1:, ...]

    mask = mask.unsqueeze(1)
    probs = probs * mask
    one_hot = one_hot * mask

    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _compute_ce_weights(
    loader: DataLoader, num_classes: int, ignore_index: int | None = None
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, _, targets in loader:
        targets = targets.view(-1)
        if ignore_index is not None:
            targets = targets[targets != ignore_index]
        if targets.numel() == 0:
            continue
        counts += torch.bincount(targets, minlength=num_classes).to(counts.dtype)
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        counts[ignore_index] = 0
    weights = torch.zeros(num_classes, dtype=torch.float64)
    nonzero = counts > 0
    if nonzero.any():
        total = counts[nonzero].sum()
        num = int(nonzero.sum().item())
        weights[nonzero] = total / (counts[nonzero] * num)
    return weights.float()


def _resolve_class_names(num_classes: int) -> List[str]:
    if num_classes == 2:
        return ["GM", "WM"]
    if num_classes == 3:
        return ["background", "GM", "WM"]
    return [f"class_{idx}" for idx in range(num_classes)]


def _slug_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _evaluate(
    model,
    loader,
    loss_fn,
    device,
    num_classes: int,
    ignore_index: int | None = None,
):
    start = time.perf_counter()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    iou_sum = torch.zeros(num_classes, dtype=torch.float64)
    iou_count = torch.zeros(num_classes, dtype=torch.float64)
    dice_sum = torch.zeros(num_classes, dtype=torch.float64)
    dice_count = torch.zeros(num_classes, dtype=torch.float64)
    with torch.no_grad():
        for inputs, m00, targets in loader:
            inputs = inputs.to(device)
            m00 = m00.to(device)
            targets = targets.to(device)
            m00_ready = _prepare_m00_tensor(m00, target_hw=inputs.shape[-2:])
            m00_present = torch.ones((inputs.shape[0],), dtype=torch.bool, device=inputs.device)
            logits = model(inputs, m00=m00_ready, m00_present=m00_present)
            loss = loss_fn(logits, targets)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            if ignore_index is None:
                mask = torch.ones_like(targets, dtype=torch.bool)
            else:
                mask = targets != ignore_index
            correct += ((preds == targets) & mask).sum().item()
            total += mask.sum().item()
            ious = _compute_iou(preds, targets, num_classes, mask=mask)
            dices = _compute_dice(preds, targets, num_classes, mask=mask)
            for cls in range(num_classes):
                if not torch.isnan(ious[cls]):
                    iou_sum[cls] += ious[cls]
                    iou_count[cls] += 1
                if not torch.isnan(dices[cls]):
                    dice_sum[cls] += dices[cls]
                    dice_count[cls] += 1
    acc = correct / max(1, total)
    iou_avg = iou_sum / torch.clamp(iou_count, min=1.0)
    mean_iou = torch.nanmean(iou_avg).item()
    dice_avg = dice_sum / torch.clamp(dice_count, min=1.0)
    mean_dice = torch.nanmean(dice_avg).item()
    elapsed = time.perf_counter() - start
    return total_loss / max(1, len(loader)), acc, iou_avg, mean_iou, dice_avg, mean_dice, elapsed


def _load_hrnet_encoder_weights(model: HRNetSegmentation, ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[SSL] Loading segmentation checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    enc_filtered = {}
    enc_state = model.encoder.state_dict()
    m00_filtered = {}
    gate_filtered = {}
    m00_state = model.m00_enc.state_dict() if getattr(model, "m00_enc", None) is not None else {}
    gate_state = model.fuse_gate.state_dict() if getattr(model, "fuse_gate", None) is not None else {}
    for key, value in state.items():
        k = key
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("encoder."):
            k2 = k[len("encoder.") :]
            if k2 in enc_state and enc_state[k2].shape == value.shape:
                enc_filtered[k2] = value
        if getattr(model, "use_m00", False):
            if k.startswith("m00_enc."):
                k2 = k[len("m00_enc.") :]
                if k2 in m00_state and m00_state[k2].shape == value.shape:
                    m00_filtered[k2] = value
            if k.startswith("fuse_gate."):
                k2 = k[len("fuse_gate.") :]
                if k2 in gate_state and gate_state[k2].shape == value.shape:
                    gate_filtered[k2] = value
    missing, unexpected = model.encoder.load_state_dict(enc_filtered, strict=False)
    print(f"[SSL] Loaded encoder tensors: {len(enc_filtered)}")
    if missing:
        print(f"[SSL] Missing keys when loading encoder: {len(missing)}")
    if unexpected:
        print(f"[SSL] Unexpected keys when loading encoder: {len(unexpected)}")
    if getattr(model, "use_m00", False):
        if not m00_filtered or not gate_filtered:
            print(
                "[SSL] WARNING: m00 fusion weights not fully found in segmentation checkpoint; "
                "m00 branch may be partially/randomly initialized."
            )
        if getattr(model, "m00_enc", None) is not None:
            model.m00_enc.load_state_dict(m00_filtered, strict=False)
        if getattr(model, "fuse_gate", None) is not None:
            model.fuse_gate.load_state_dict(gate_filtered, strict=False)


def _append_metrics(
    log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    acc: float,
    mean_iou: float,
    iou_avg: torch.Tensor,
    mean_dice: float,
    dice_avg: torch.Tensor,
    class_names: List[str],
) -> None:
    if not log_path.exists():
        class_cols = [f"iou_{_slug_class_name(name)}" for name in class_names]
        dice_cols = [f"dice_{_slug_class_name(name)}" for name in class_names]
        log_path.write_text(
            "epoch,train_loss,val_loss,acc,mean_iou,mean_dice,"
            + ",".join(class_cols)
            + ","
            + ",".join(dice_cols)
            + "\n"
        )
    with log_path.open("a") as f:
        iou_values = []
        for idx in range(len(class_names)):
            val = float(iou_avg[idx].item())
            iou_values.append("nan" if math.isnan(val) else f"{val:.6f}")
        dice_values = []
        for idx in range(len(class_names)):
            val = float(dice_avg[idx].item())
            dice_values.append("nan" if math.isnan(val) else f"{val:.6f}")
        f.write(
            f"{epoch},{train_loss:.6f},{val_loss:.6f},{acc:.6f},{mean_iou:.6f},{mean_dice:.6f},"
            + ",".join(iou_values)
            + ","
            + ",".join(dice_values)
            + "\n"
        )


def _write_test_metrics(
    log_path: Path,
    acc: float,
    mean_iou: float,
    iou_avg: torch.Tensor,
    mean_dice: float,
    dice_avg: torch.Tensor,
    class_names: List[str],
) -> None:
    if not log_path.exists():
        class_cols = [f"iou_{_slug_class_name(name)}" for name in class_names]
        dice_cols = [f"dice_{_slug_class_name(name)}" for name in class_names]
        log_path.write_text(
            "acc,mean_iou,mean_dice," + ",".join(class_cols) + "," + ",".join(dice_cols) + "\n"
        )
    iou_values = []
    for idx in range(len(class_names)):
        val = float(iou_avg[idx].item())
        iou_values.append("nan" if math.isnan(val) else f"{val:.6f}")
    dice_values = []
    for idx in range(len(class_names)):
        val = float(dice_avg[idx].item())
        dice_values.append("nan" if math.isnan(val) else f"{val:.6f}")
    with log_path.open("a") as f:
        f.write(
            f"{acc:.6f},{mean_iou:.6f},{mean_dice:.6f},"
            + ",".join(iou_values)
            + ","
            + ",".join(dice_values)
            + "\n"
        )


def _few_shot_to_key(few_shot_pct: int | None) -> int:
    return 100 if few_shot_pct is None else int(few_shot_pct)


def _parse_int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _load_completed_run_keys(
    summary_path: Path,
    nested: bool,
) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    if not summary_path.exists():
        return keys
    fold_field = "pair_index" if nested else "fold"
    with summary_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_name = (row.get("run") or "").strip()
            if not run_name:
                continue
            fold_idx = _parse_int(row.get(fold_field), default=-1)
            if fold_idx < 0:
                continue
            few_shot_pct = _parse_int(row.get("few_shot_pct"), default=100)
            keys.add((run_name, fold_idx, few_shot_pct))
    return keys


def _compute_majority_baseline(
    loader: DataLoader, num_classes: int, ignore_index: int | None = None
) -> Tuple[int, float, List[float], float]:
    class_counts = torch.zeros(num_classes, dtype=torch.int64)
    for _, _, targets in loader:
        targets = targets.view(-1)
        if ignore_index is not None:
            targets = targets[targets != ignore_index]
        if targets.numel() == 0:
            continue
        class_counts += torch.bincount(targets, minlength=num_classes)
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        class_counts[ignore_index] = 0
    majority_class = int(torch.argmax(class_counts).item())

    correct = 0
    total = 0
    iou_sum = torch.zeros(num_classes, dtype=torch.float64)
    iou_count = torch.zeros(num_classes, dtype=torch.float64)
    for _, _, targets in loader:
        preds = torch.full_like(targets, majority_class)
        if ignore_index is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            mask = targets != ignore_index
        correct += ((preds == targets) & mask).sum().item()
        total += mask.sum().item()
        ious = _compute_iou(preds, targets, num_classes, mask=mask)
        for cls in range(num_classes):
            if not torch.isnan(ious[cls]):
                iou_sum[cls] += ious[cls]
                iou_count[cls] += 1
    acc = correct / max(1, total)
    iou_avg = iou_sum / torch.clamp(iou_count, min=1.0)
    mean_iou = torch.nanmean(iou_avg).item()
    return majority_class, acc, [float(v) for v in iou_avg.tolist()], mean_iou


def _fold_dir_for_run(
    outdir: Path, run_name: str, fold_idx: int, few_shot_pct: int | None
) -> Path:
    run_dir = outdir / run_name
    if few_shot_pct is not None:
        run_dir = run_dir / f"few_shot_{few_shot_pct}pct"
    return run_dir / f"fold_{fold_idx}"


def _select_few_shot_subset(
    dataset: PolambrimetryFullDataset,
    few_shot_pct: int,
    seed: int,
) -> tuple[Subset, int, int]:
    total = len(dataset)
    if total <= 0:
        raise ValueError("Training dataset is empty; cannot apply few-shot sampling.")
    keep = int(round(total * (few_shot_pct / 100.0)))
    keep = max(1, min(total, keep))
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    picked = sorted(indices[:keep])
    return Subset(dataset, picked), keep, total


def _run_fold(
    fold_idx: int,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    cfg: TrainConfig,
    outdir: Path,
    device: str,
    run_name: str,
    seg_encoder_checkpoint: Path | None,
    baseline: Tuple[int, float, List[float], float] | None = None,
    nested_info: Optional[Dict[str, object]] = None,
    few_shot_pct: int | None = None,
):
    class_names = _resolve_class_names(cfg.num_classes)
    ignore_index = IGNORE_INDEX if cfg.num_classes == 2 else 0
    train_ds_full = PolambrimetryFullDataset(
        train_ids,
        augment=True,
        rotation_prob=cfg.rotation_prob,
        flip_prob=cfg.flip_prob,
        rotation_deg=cfg.rotation_deg,
        patch_size=None,
        center_crop_size=cfg.center_crop_size,
        num_classes=cfg.num_classes,
        ignore_index=ignore_index,
        use_m00=True,
        m00_log_input=False,
        m00_log_eps=cfg.m00_log_eps,
    )
    train_ds = train_ds_full
    few_shot_kept = len(train_ds_full)
    few_shot_total = len(train_ds_full)
    if few_shot_pct is not None:
        few_shot_seed = cfg.seed + (fold_idx + 1) * 1009 + few_shot_pct * 17
        train_ds, few_shot_kept, few_shot_total = _select_few_shot_subset(
            train_ds_full, few_shot_pct=few_shot_pct, seed=few_shot_seed
        )
        print(
            f"[fold {fold_idx}] few-shot {few_shot_pct}% -> "
            f"{few_shot_kept}/{few_shot_total} training samples"
        )
    val_ds = None
    if val_ids:
        val_ds = PolambrimetryFullDataset(
            val_ids,
            augment=False,
            patch_size=None,
            center_crop_size=cfg.center_crop_size,
            num_classes=cfg.num_classes,
            ignore_index=ignore_index,
            use_m00=True,
            m00_log_input=False,
            m00_log_eps=cfg.m00_log_eps,
        )
    test_ds = PolambrimetryFullDataset(
        test_ids,
        augment=False,
        patch_size=None,
        center_crop_size=cfg.center_crop_size,
        num_classes=cfg.num_classes,
        ignore_index=ignore_index,
        use_m00=True,
        m00_log_input=False,
        m00_log_eps=cfg.m00_log_eps,
    )

    base_seed = cfg.seed + fold_idx + 1
    generator = torch.Generator()
    if cfg.deterministic:
        generator.manual_seed(base_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=(lambda wid: _seed_worker(wid, base_seed)) if cfg.deterministic else None,
        generator=generator if cfg.deterministic else None,
    )
    weight_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=(lambda wid: _seed_worker(wid, base_seed)) if cfg.deterministic else None,
        generator=generator if cfg.deterministic else None,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            worker_init_fn=(lambda wid: _seed_worker(wid, base_seed)) if cfg.deterministic else None,
            generator=generator if cfg.deterministic else None,
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=(lambda wid: _seed_worker(wid, base_seed)) if cfg.deterministic else None,
        generator=generator if cfg.deterministic else None,
    )
    if baseline is None:
        baseline = _compute_majority_baseline(test_loader, cfg.num_classes, ignore_index=ignore_index)
    baseline_majority_class, baseline_acc, baseline_iou, baseline_mean_iou = baseline
    print(
        f"[fold {fold_idx}] baseline majority class={baseline_majority_class} "
        f"acc={baseline_acc:.4f} mean_iou={baseline_mean_iou:.4f}"
    )

    model = HRNetSegmentation(
        variant=cfg.encoder_variant,
        in_channels=16,
        num_classes=cfg.num_classes,
        pretrained=False,
        head_upsample="bilinear",
        use_m00=True,
        m00_log_input=False,
    ).to(device)
    if seg_encoder_checkpoint is not None:
        _load_hrnet_encoder_weights(model, seg_encoder_checkpoint)
    ce_weights = _compute_ce_weights(weight_loader, cfg.num_classes, ignore_index=ignore_index).to(device)
    ce_criterion = torch.nn.CrossEntropyLoss(weight=ce_weights, ignore_index=ignore_index)
    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = ce_criterion(logits, targets)
        dice_loss = _dice_loss(logits, targets, cfg.num_classes, ignore_index=ignore_index)
        return 0.5 * ce_loss + 0.5 * dice_loss
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_plateau_factor,
        patience=cfg.lr_plateau_patience,
        min_lr=cfg.lr_plateau_min_lr,
    )
    fold_dir = _fold_dir_for_run(outdir, run_name, fold_idx, few_shot_pct)
    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = fold_dir / "metrics.csv"

    if nested_info is not None:
        split_info = dict(nested_info)
        split_info["few_shot_pct"] = few_shot_pct
        split_info["few_shot_kept"] = few_shot_kept
        split_info["few_shot_total"] = few_shot_total
        _write_split_info(fold_dir, split_info)
        test_label = Path(test_ids[0]).name if test_ids else "n/a"
        val_label = Path(val_ids[0]).name if val_ids else "n/a"
        print(
            f"[fold {fold_idx}] nested outer={nested_info.get('outer_fold')} "
            f"test={test_label} val={val_label} train={len(train_ids)}"
        )

    few_shot_label = "full" if few_shot_pct is None else f"{few_shot_pct}%"
    print(f"[TRAIN] Training on device: {device}, {run_name} fold {fold_idx} few-shot={few_shot_label}")

    best_acc = 0.0
    best_mean_iou = 0.0
    best_mean_dice = 0.0
    best_iou = [float("nan")] * cfg.num_classes
    best_dice = [float("nan")] * cfg.num_classes
    best_val_loss = float("inf")
    epochs_no_improve = 0
    eval_loader = val_loader if val_loader is not None else test_loader
    select_by_dice = val_loader is not None
    early_stop_patience = cfg.nested_early_stop_patience
    use_early_stop = val_loader is not None and early_stop_patience > 0
    best_val_metric_for_stop = float("-inf") if select_by_dice else float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    best_val_mean_iou = 0.0
    best_val_mean_dice = float("-inf")
    best_val_iou = [float("nan")] * cfg.num_classes
    best_val_dice = [float("nan")] * cfg.num_classes
    for epoch in range(cfg.epochs):
        train_loss, data_time, step_time = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )
        val_loss, val_acc, iou_avg, mean_iou, dice_avg, mean_dice, eval_time = _evaluate(
            model,
            eval_loader,
            loss_fn,
            device,
            cfg.num_classes,
            ignore_index=ignore_index,
        )
        scheduler.step(val_loss)
        _append_metrics(
            metrics_path,
            epoch + 1,
            train_loss,
            val_loss,
            val_acc,
            mean_iou,
            iou_avg,
            mean_dice,
            dice_avg,
            class_names,
        )
        print(
            f"[fold {fold_idx}] epoch {epoch+1}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"mean_iou={mean_iou:.4f} iou={iou_avg.tolist()} "
        )
        if select_by_dice:
            if mean_dice > best_val_mean_dice:
                best_val_mean_dice = mean_dice
                best_epoch = epoch + 1
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_mean_iou = mean_iou
                best_val_iou = [float(v) for v in iou_avg.tolist()]
                best_val_dice = [float(v) for v in dice_avg.tolist()]
                best_acc = val_acc
                best_mean_iou = mean_iou
                best_mean_dice = mean_dice
                best_iou = best_val_iou[:]
                best_dice = best_val_dice[:]
                torch.save(model.state_dict(), fold_dir / "best.pt")
        else:
            for cls in range(cfg.num_classes):
                val = float(iou_avg[cls].item())
                if math.isnan(val):
                    continue
                if math.isnan(best_iou[cls]) or val > best_iou[cls]:
                    best_iou[cls] = val
            for cls in range(cfg.num_classes):
                val = float(dice_avg[cls].item())
                if math.isnan(val):
                    continue
                if math.isnan(best_dice[cls]) or val > best_dice[cls]:
                    best_dice[cls] = val
            if val_acc > best_acc:
                best_acc = val_acc
                best_mean_iou = mean_iou
                best_mean_dice = mean_dice
                best_epoch = epoch + 1
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_mean_iou = mean_iou
                best_val_mean_dice = mean_dice
                best_val_iou = [float(v) for v in iou_avg.tolist()]
                best_val_dice = [float(v) for v in dice_avg.tolist()]
                torch.save(model.state_dict(), fold_dir / "best.pt")
        if use_early_stop:
            improved = False
            if select_by_dice:
                if mean_dice > best_val_metric_for_stop + cfg.early_stop_min_delta:
                    best_val_metric_for_stop = mean_dice
                    improved = True
            else:
                if val_loss < best_val_metric_for_stop - cfg.early_stop_min_delta:
                    best_val_metric_for_stop = val_loss
                    improved = True
            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(
                        f"[fold {fold_idx}] early stopping at epoch {epoch+1} "
                        f"(no improvement for {early_stop_patience} epochs)"
                    )
                    break

    torch.save(model.state_dict(), fold_dir / "last.pt")
    # Final test evaluation on held-out specimen(s)
    best_path = fold_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_dir = fold_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_loss, test_acc, test_iou, test_mean_iou, test_dice, test_mean_dice, _ = _evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        cfg.num_classes,
        ignore_index=ignore_index,
    )
    _write_test_metrics(
        test_dir / "metrics.csv",
        test_acc,
        test_mean_iou,
        test_iou,
        test_mean_dice,
        test_dice,
        class_names,
    )
    print(
        f"[fold {fold_idx}] test_loss={test_loss:.4f} "
        f"test_acc={test_acc:.4f} test_mean_iou={test_mean_iou:.4f}"
    )
    if nested_info is not None:
        outer_fold = int(nested_info.get("outer_fold", -1))
        _append_nested_summary(
            outdir / "nested_summary.csv",
            run_name,
            fold_idx,
            outer_fold,
            few_shot_pct,
            list(nested_info.get("train_ids", [])),
            list(nested_info.get("val_ids", [])),
            list(nested_info.get("test_ids", [])),
            best_epoch,
            best_val_loss,
            best_val_acc,
            best_val_mean_iou,
            best_val_mean_dice,
            best_val_iou,
            best_val_dice,
            test_loss,
            test_acc,
            test_mean_iou,
            test_mean_dice,
            test_iou,
            test_dice,
            class_names,
            baseline_majority_class,
            baseline_acc,
            baseline_mean_iou,
            baseline_iou,
            seg_encoder_checkpoint,
        )
    return baseline


def _write_split_info(fold_dir: Path, info: Dict[str, object]) -> None:
    path = fold_dir / "split.json"
    payload = _json_safe(info)
    path.write_text(json.dumps(payload, indent=2))


def _append_nested_summary(
    summary_path: Path,
    run_name: str,
    pair_index: int,
    outer_fold: int,
    few_shot_pct: int | None,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    best_epoch: int,
    best_val_loss: float,
    best_val_acc: float,
    best_val_mean_iou: float,
    best_val_mean_dice: float,
    best_val_iou: List[float],
    best_val_dice: List[float],
    test_loss: float,
    test_acc: float,
    test_mean_iou: float,
    test_mean_dice: float,
    test_iou: torch.Tensor,
    test_dice: torch.Tensor,
    class_names: List[str],
    baseline_majority_class: int,
    baseline_acc: float,
    baseline_mean_iou: float,
    baseline_iou: List[float],
    seg_ckpt_path: Path | None,
) -> None:
    if not summary_path.exists():
        val_iou_cols = [f"best_val_iou_{_slug_class_name(name)}" for name in class_names]
        val_dice_cols = [f"best_val_dice_{_slug_class_name(name)}" for name in class_names]
        test_iou_cols = [f"test_iou_{_slug_class_name(name)}" for name in class_names]
        test_dice_cols = [f"test_dice_{_slug_class_name(name)}" for name in class_names]
        baseline_cols = [f"baseline_iou_{_slug_class_name(name)}" for name in class_names]
        summary_path.write_text(
            "run,pair_index,outer_fold,few_shot_pct,test_specimen,val_specimen,train_specimens,"
            "best_epoch,best_val_loss,best_val_acc,best_val_mean_iou,best_val_mean_dice,"
            + ",".join(val_iou_cols)
            + ","
            + ",".join(val_dice_cols)
            + ",test_loss,test_acc,test_mean_iou,test_mean_dice,"
            + ",".join(test_iou_cols)
            + ","
            + ",".join(test_dice_cols)
            + ",baseline_majority_class,baseline_acc,baseline_mean_iou,"
            + ",".join(baseline_cols)
            + ",seg_encoder_checkpoint\n"
        )

    def _join_ids(ids: Iterable[str]) -> str:
        return "|".join(ids)

    test_spec = test_ids[0] if test_ids else ""
    val_spec = val_ids[0] if val_ids else ""
    train_spec = _join_ids(train_ids)
    few_shot_str = "100" if few_shot_pct is None else str(few_shot_pct)
    seg_ckpt_str = str(seg_ckpt_path) if seg_ckpt_path is not None else ""

    val_iou_values = []
    for val in best_val_iou:
        val_iou_values.append("nan" if math.isnan(val) else f"{val:.6f}")
    val_dice_values = []
    for val in best_val_dice:
        val_dice_values.append("nan" if math.isnan(val) else f"{val:.6f}")

    test_iou_values = []
    for idx in range(len(class_names)):
        val = float(test_iou[idx].item())
        test_iou_values.append("nan" if math.isnan(val) else f"{val:.6f}")
    test_dice_values = []
    for idx in range(len(class_names)):
        val = float(test_dice[idx].item())
        test_dice_values.append("nan" if math.isnan(val) else f"{val:.6f}")

    baseline_iou_values = []
    for val in baseline_iou:
        baseline_iou_values.append("nan" if math.isnan(val) else f"{val:.6f}")

    with summary_path.open("a") as f:
        f.write(
            f"{run_name},{pair_index},{outer_fold},{few_shot_str},{test_spec},{val_spec},{train_spec},"
            f"{best_epoch},{best_val_loss:.6f},{best_val_acc:.6f},{best_val_mean_iou:.6f},"
            f"{best_val_mean_dice:.6f},"
            + ",".join(val_iou_values)
            + ","
            + ",".join(val_dice_values)
            + f",{test_loss:.6f},{test_acc:.6f},{test_mean_iou:.6f},{test_mean_dice:.6f},"
            + ",".join(test_iou_values)
            + ","
            + ",".join(test_dice_values)
            + f",{baseline_majority_class},{baseline_acc:.6f},{baseline_mean_iou:.6f},"
            + ",".join(baseline_iou_values)
            + f",{seg_ckpt_str}\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the MuellerPT PoLambRimetry few-shot table"
    )
    parser.add_argument("--fold-index", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(os.environ.get("MUELLERPT_OUTPUT_ROOT", str(Path(__file__).resolve().parents[2] / "outputs"))) / "results" / "polambrimetry",
    )
    parser.add_argument("--seg-encoder-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--few-shot-percentages",
        type=int,
        nargs="+",
        default=None,
        help="Few-shot training percentages (sample-level), e.g. --few-shot-percentages 1 5 25 50",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Exact run directory name under --outdir (use this to continue an existing run).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume an existing run by skipping entries already present in summary CSVs.",
    )
    args = parser.parse_args()

    cfg = _build_config(args)
    run_id = args.run_id if args.run_id else _make_run_id("paper_nested_cv")
    _run_training(cfg, args, args.outdir, run_id, args.fold_index)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
