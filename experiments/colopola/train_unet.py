from __future__ import annotations

import argparse
import csv
import json
import os
import random
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import TrainConfig, resolve_device
from dataset import (
    ColopolaSample,
    ColopolaClassificationDataset,
    load_colopola_samples,
    split_colopola_samples,
    summarize_class_counts,
)
from model import HRNetClassifier, load_encoder_from_checkpoint


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_run_config(run_dir: Path, cfg: TrainConfig, args: argparse.Namespace) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": _json_safe(asdict(cfg)),
        "args": _json_safe(vars(args)),
    }
    (run_dir / "config_used.json").write_text(json.dumps(payload, indent=2))

    src = Path(__file__).resolve().parent / "config.py"
    dst = run_dir / "config_used.py"
    if src.exists():
        dst.write_text(src.read_text())


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
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            print(f"[WARN] Deterministic algorithms not fully enabled: {exc}")


def _seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _normalize_few_shot_percentages(values: List[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        pct = int(value)
        if pct <= 0 or pct > 100:
            raise ValueError(f"few-shot percentage must be in [1, 100], got {pct}")
        if pct not in seen:
            out.append(pct)
            seen.add(pct)
    return out


def _normalize_split_seeds(values: Optional[List[int]]) -> Optional[List[int]]:
    if values is None:
        return None
    out: List[int] = []
    seen = set()
    for value in values:
        seed = int(value)
        if seed < 0:
            raise ValueError(f"split seed must be >= 0, got {seed}")
        if seed not in seen:
            out.append(seed)
            seen.add(seed)
    if not out:
        raise ValueError("split_seeds was provided but empty after normalization.")
    return out


def _resolve_split_seeds(cfg: TrainConfig) -> List[int]:
    seeds = _normalize_split_seeds(cfg.split_seeds)
    if seeds is not None:
        return seeds
    return [int(cfg.split_seed)]


def _validate_config(cfg: TrainConfig) -> None:
    total_split = cfg.train_fraction + cfg.val_fraction + cfg.test_fraction
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(
            "train_fraction + val_fraction + test_fraction must equal 1.0 "
            f"(got {total_split:.6f})."
        )
    cfg.few_shot_percentages = _normalize_few_shot_percentages(cfg.few_shot_percentages)
    cfg.split_seeds = _normalize_split_seeds(cfg.split_seeds)
    if not cfg.few_shot_percentages:
        raise ValueError("At least one few-shot percentage is required.")
    if cfg.encoder_checkpoint is None:
        raise ValueError("MuellerPT checkpoint is required for the paper comparison.")


def _build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()

    if args.dataset_root is not None:
        cfg.dataset_root = Path(args.dataset_root)
    if args.output_dir is not None:
        cfg.output_dir = Path(args.output_dir)
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.encoder_checkpoint is not None:
        cfg.encoder_checkpoint = args.encoder_checkpoint

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.split_seeds is not None:
        cfg.split_seeds = [int(v) for v in args.split_seeds]
    if args.few_shot_percentages is not None:
        cfg.few_shot_percentages = [int(v) for v in args.few_shot_percentages]

    cfg.device = resolve_device(args.device if args.device is not None else cfg.device)

    _validate_config(cfg)

    return cfg


def _write_sweep_progress(sweep_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    out_path = sweep_dir / "sweep_trials.csv"
    fieldnames = ["seed", "run_id", "run_dir"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def _compute_class_weights(class_counts: Dict[int, int]) -> torch.Tensor:
    total = float(sum(class_counts.values()))
    weights = torch.ones(2, dtype=torch.float32)
    for cls in (0, 1):
        count = class_counts.get(cls, 0)
        if count > 0:
            weights[cls] = total / (2.0 * float(count))
    return weights


def _build_loader(
    cfg: TrainConfig,
    samples: List[ColopolaSample],
    augment: bool,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    dataset = ColopolaClassificationDataset(
        samples,
        augment=augment,
        rotation_prob=cfg.rotation_prob,
        flip_prob=cfg.flip_prob,
        rotation_deg=cfg.rotation_deg,
        center_crop_size=cfg.center_crop_size,
        pad_multiple=cfg.pad_multiple,
    )

    pin_memory = torch.cuda.is_available()
    base_seed = cfg.split_seed + seed_offset
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(base_seed)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=lambda wid, bs=base_seed: _seed_worker(wid, bs),
        generator=generator,
        drop_last=False,
    )


def _select_few_shot_samples(
    train_samples: List[ColopolaSample],
    few_shot_pct: int,
    seed: int,
) -> List[ColopolaSample]:
    if few_shot_pct <= 0 or few_shot_pct > 100:
        raise ValueError(f"few-shot percentage must be in [1, 100], got {few_shot_pct}")
    if few_shot_pct == 100:
        return list(train_samples)

    by_label: Dict[int, List[ColopolaSample]] = {0: [], 1: []}
    for sample in train_samples:
        by_label[sample.label].append(sample)

    selected: List[ColopolaSample] = []
    for label in (0, 1):
        label_samples = list(by_label[label])
        rng = random.Random(seed + 1000 * label + few_shot_pct)
        rng.shuffle(label_samples)
        if not label_samples:
            continue
        k = int(round(len(label_samples) * (few_shot_pct / 100.0)))
        k = max(1, min(k, len(label_samples)))
        selected.extend(label_samples[:k])

    rng = random.Random(seed + few_shot_pct)
    rng.shuffle(selected)
    return selected


def _build_loaders(
    cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, Dict[str, List], Dict[str, Dict[int, int]]]:
    samples = load_colopola_samples(cfg.dataset_root)
    train_samples, val_samples, test_samples = split_colopola_samples(
        samples,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        seed=cfg.split_seed,
    )

    if not val_samples:
        val_samples = list(train_samples)
        print("[WARN] Empty validation split. Reusing training split for validation.")
    if not test_samples:
        test_samples = list(val_samples)
        print("[WARN] Empty test split. Reusing validation split for testing.")

    split_counts = {
        "train": summarize_class_counts(train_samples),
        "val": summarize_class_counts(val_samples),
        "test": summarize_class_counts(test_samples),
    }
    print(
        "[SPLIT] train={0} val={1} test={2}".format(
            len(train_samples), len(val_samples), len(test_samples)
        )
    )
    print(
        "[SPLIT] class counts train={0} val={1} test={2}".format(
            split_counts["train"], split_counts["val"], split_counts["test"]
        )
    )

    train_loader = _build_loader(
        cfg,
        train_samples,
        augment=True,
        shuffle=True,
        seed_offset=0,
    )
    val_loader = _build_loader(
        cfg,
        val_samples,
        augment=False,
        shuffle=False,
        seed_offset=1000,
    )
    test_loader = _build_loader(
        cfg,
        test_samples,
        augment=False,
        shuffle=False,
        seed_offset=2000,
    )

    split_samples = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }
    return train_loader, val_loader, test_loader, split_samples, split_counts


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    grad_clip_max_norm: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    tp = tn = fp = fn = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(inputs)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)

            total_loss += float(loss.item())
            total += int(targets.numel())
            correct += int((preds == targets).sum().item())

            tp += int(((preds == 1) & (targets == 1)).sum().item())
            tn += int(((preds == 0) & (targets == 0)).sum().item())
            fp += int(((preds == 1) & (targets == 0)).sum().item())
            fn += int(((preds == 0) & (targets == 1)).sum().item())

    acc = correct / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_acc = 0.5 * (recall + specificity)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)

    # Per-class metrics: class 0 = non-cancer, class 1 = cancer.
    precision_non_cancer = tn / max(1, tn + fn)
    recall_non_cancer = tn / max(1, tn + fp)
    f1_non_cancer = (
        2.0 * precision_non_cancer * recall_non_cancer
        / max(1e-12, precision_non_cancer + recall_non_cancer)
    )
    support_non_cancer = tn + fp

    precision_cancer = precision
    recall_cancer = recall
    f1_cancer = f1
    support_cancer = tp + fn

    return {
        "loss": total_loss / max(1, len(loader)),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_acc": balanced_acc,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "precision_non_cancer": precision_non_cancer,
        "recall_non_cancer": recall_non_cancer,
        "f1_non_cancer": f1_non_cancer,
        "support_non_cancer": float(support_non_cancer),
        "precision_cancer": precision_cancer,
        "recall_cancer": recall_cancer,
        "f1_cancer": f1_cancer,
        "support_cancer": float(support_cancer),
    }


def _append_epoch_metrics(
    metrics_path: Path,
    epoch: int,
    train_loss: float,
    val_metrics: Dict[str, float],
    lr: float,
) -> None:
    header = (
        "epoch,train_loss,val_loss,val_acc,val_precision,val_recall,"
        "val_specificity,val_balanced_acc,val_f1,lr\n"
    )
    if not metrics_path.exists():
        metrics_path.write_text(header)

    with metrics_path.open("a") as f:
        f.write(
            f"{epoch},{train_loss:.6f},{val_metrics['loss']:.6f},{val_metrics['acc']:.6f},"
            f"{val_metrics['precision']:.6f},{val_metrics['recall']:.6f},"
            f"{val_metrics['specificity']:.6f},{val_metrics['balanced_acc']:.6f},"
            f"{val_metrics['f1']:.6f},{lr:.8f}\n"
        )


def _write_test_metrics(metrics_path: Path, test_metrics: Dict[str, float]) -> None:
    header = (
        "loss,acc,precision,recall,specificity,balanced_acc,f1,tp,tn,fp,fn,"
        "precision_non_cancer,recall_non_cancer,f1_non_cancer,support_non_cancer,"
        "precision_cancer,recall_cancer,f1_cancer,support_cancer\n"
    )
    metrics_path.write_text(header)
    with metrics_path.open("a") as f:
        f.write(
            f"{test_metrics['loss']:.6f},{test_metrics['acc']:.6f},"
            f"{test_metrics['precision']:.6f},{test_metrics['recall']:.6f},"
            f"{test_metrics['specificity']:.6f},{test_metrics['balanced_acc']:.6f},"
            f"{test_metrics['f1']:.6f},{int(test_metrics['tp'])},"
            f"{int(test_metrics['tn'])},{int(test_metrics['fp'])},{int(test_metrics['fn'])},"
            f"{test_metrics['precision_non_cancer']:.6f},{test_metrics['recall_non_cancer']:.6f},"
            f"{test_metrics['f1_non_cancer']:.6f},{int(test_metrics['support_non_cancer'])},"
            f"{test_metrics['precision_cancer']:.6f},{test_metrics['recall_cancer']:.6f},"
            f"{test_metrics['f1_cancer']:.6f},{int(test_metrics['support_cancer'])}\n"
        )


def _write_aggregated_results(run_dir: Path, rows: List[Dict[str, object]]) -> Path:
    out_path = run_dir / "aggregated_results.csv"
    fieldnames = [
        "run_id",
        "phase",
        "few_shot_pct",
        "mode",
        "train_samples",
        "train_normal",
        "train_cancer",
        "best_epoch",
        "best_val_loss",
        "best_val_acc",
        "best_val_f1",
        "test_loss",
        "test_acc",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_specificity",
        "test_balanced_acc",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def _append_global_aggregated_results(
    output_dir: Path,
    rows: List[Dict[str, object]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "aggregated_results_all_runs.csv"
    fieldnames = [
        "run_id",
        "phase",
        "few_shot_pct",
        "mode",
        "train_samples",
        "train_normal",
        "train_cancer",
        "best_epoch",
        "best_val_loss",
        "best_val_acc",
        "best_val_f1",
        "test_loss",
        "test_acc",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_specificity",
        "test_balanced_acc",
    ]
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def _write_test_metrics_by_mode(run_dir: Path, rows: List[Dict[str, object]]) -> Path:
    out_path = run_dir / "test_metrics_by_mode.csv"
    fieldnames = [
        "run_id",
        "phase",
        "few_shot_pct",
        "mode",
        "test_loss",
        "test_acc",
        "test_balanced_acc",
        "test_precision_non_cancer",
        "test_recall_non_cancer",
        "test_f1_non_cancer",
        "test_support_non_cancer",
        "test_precision_cancer",
        "test_recall_cancer",
        "test_f1_cancer",
        "test_support_cancer",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def _run_experiment(
    run_name: str,
    cfg: TrainConfig,
    run_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    encoder_checkpoint: Optional[str],
) -> tuple[Path, Path, Dict[str, float], Dict[str, float]]:
    out_dir = run_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = HRNetClassifier(
        variant=cfg.encoder_variant,
        in_channels=16,
        num_classes=cfg.num_classes,
        pretrained=False,
    ).to(cfg.device)

    if encoder_checkpoint:
        loaded, missing, unexpected = load_encoder_from_checkpoint(
            model, Path(encoder_checkpoint)
        )
        print(
            f"[{run_name}] loaded encoder tensors={loaded} missing={missing} unexpected={unexpected}"
        )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.device))

    metrics_path = out_dir / "metrics.csv"
    best_ckpt_path = out_dir / "best.pt"

    best_val_f1 = float("-inf")
    best_epoch = 0
    best_val_metrics: Dict[str, float] = {
        "loss": float("nan"),
        "acc": float("nan"),
        "f1": float("nan"),
    }
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            cfg.device,
            grad_clip_max_norm=cfg.grad_clip_max_norm,
        )
        val_metrics = _evaluate(model, val_loader, criterion, cfg.device)

        _append_epoch_metrics(
            metrics_path,
            epoch,
            train_loss,
            val_metrics,
            lr=optimizer.param_groups[0]["lr"],
        )

        improved = val_metrics["f1"] > (best_val_f1 + cfg.early_stop_min_delta)
        if improved:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_metrics": val_metrics,
                },
                best_ckpt_path,
            )
        else:
            no_improve += 1

        print(
            f"[{run_name}] epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if no_improve >= cfg.early_stop_patience:
            print(
                f"[{run_name}] Early stop at epoch {epoch} "
                f"(best epoch {best_epoch}, best val_f1={best_val_f1:.4f})."
            )
            break

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])

    test_metrics = _evaluate(model, test_loader, criterion, cfg.device)
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    _write_test_metrics(test_dir / "metrics.csv", test_metrics)

    print(
        f"[{run_name}] test_acc={test_metrics['acc']:.4f} test_f1={test_metrics['f1']:.4f} "
        f"balanced_acc={test_metrics['balanced_acc']:.4f}"
    )

    summary = {
        "best_epoch": float(best_epoch),
        "best_val_loss": float(best_val_metrics.get("loss", float("nan"))),
        "best_val_acc": float(best_val_metrics.get("acc", float("nan"))),
        "best_val_f1": float(best_val_metrics.get("f1", float("nan"))),
        "test_loss": float(test_metrics.get("loss", float("nan"))),
        "test_acc": float(test_metrics.get("acc", float("nan"))),
        "test_f1": float(test_metrics.get("f1", float("nan"))),
        "test_precision": float(test_metrics.get("precision", float("nan"))),
        "test_recall": float(test_metrics.get("recall", float("nan"))),
        "test_specificity": float(test_metrics.get("specificity", float("nan"))),
        "test_balanced_acc": float(test_metrics.get("balanced_acc", float("nan"))),
    }
    return metrics_path, best_ckpt_path, summary, test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the MuellerPT ColoPola few-shot table."
    )

    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--encoder-checkpoint", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--few-shot-percentages", type=int, nargs="+", default=None)

    return parser.parse_args()


def _run_single_config(
    cfg: TrainConfig,
    args: argparse.Namespace,
    run_id_override: Optional[str] = None,
) -> Path:
    run_id = run_id_override or cfg.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_colopola"
    run_dir = cfg.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_run_config(run_dir, cfg, args)
    _set_determinism(cfg.split_seed, cfg.deterministic)

    _, val_loader, test_loader, split_samples, _ = _build_loaders(cfg)
    base_train_samples = list(split_samples["train"])
    aggregated_rows: List[Dict[str, object]] = []
    test_mode_rows: List[Dict[str, object]] = []

    few_shot_percentages = list(cfg.few_shot_percentages)
    print(f"[RUN] paper few-shot percentages: {few_shot_percentages}")

    use_phase_subdirs = len(few_shot_percentages) > 1

    for few_shot_pct in few_shot_percentages:
        if few_shot_pct == 100:
            train_samples = list(base_train_samples)
            phase_name = "full_data"
        else:
            train_samples = _select_few_shot_samples(
                base_train_samples,
                few_shot_pct=few_shot_pct,
                seed=cfg.split_seed,
            )
            phase_name = f"few_shot_{few_shot_pct:03d}"

        phase_dir = run_dir / phase_name if use_phase_subdirs else run_dir
        phase_dir.mkdir(parents=True, exist_ok=True)

        train_counts = summarize_class_counts(train_samples)
        class_weights = _compute_class_weights(train_counts)
        print(
            f"[RUN] phase={phase_name} train_samples={len(train_samples)} "
            f"class_counts={train_counts} class_weights={class_weights.tolist()}"
        )

        train_loader = _build_loader(
            cfg,
            train_samples,
            augment=True,
            shuffle=True,
            seed_offset=3000 + few_shot_pct,
        )

        ssl_metrics_path: Optional[Path] = None
        random_metrics_path: Optional[Path] = None

        if cfg.encoder_checkpoint:
            ssl_metrics_path, _, ssl_summary, ssl_test_metrics = _run_experiment(
                run_name="ssl_pretrained",
                cfg=cfg,
                run_dir=phase_dir,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_weights=class_weights,
                encoder_checkpoint=cfg.encoder_checkpoint,
            )
            aggregated_rows.append(
                {
                    "run_id": run_id,
                    "phase": phase_name,
                    "few_shot_pct": few_shot_pct,
                    "mode": "ssl_pretrained",
                    "train_samples": len(train_samples),
                    "train_normal": train_counts.get(0, 0),
                    "train_cancer": train_counts.get(1, 0),
                    **ssl_summary,
                }
            )
            test_mode_rows.append(
                {
                    "run_id": run_id,
                    "phase": phase_name,
                    "few_shot_pct": few_shot_pct,
                    "mode": "ssl_pretrained",
                    "test_loss": float(ssl_test_metrics.get("loss", float("nan"))),
                    "test_acc": float(ssl_test_metrics.get("acc", float("nan"))),
                    "test_balanced_acc": float(
                        ssl_test_metrics.get("balanced_acc", float("nan"))
                    ),
                    "test_precision_non_cancer": float(
                        ssl_test_metrics.get("precision_non_cancer", float("nan"))
                    ),
                    "test_recall_non_cancer": float(
                        ssl_test_metrics.get("recall_non_cancer", float("nan"))
                    ),
                    "test_f1_non_cancer": float(
                        ssl_test_metrics.get("f1_non_cancer", float("nan"))
                    ),
                    "test_support_non_cancer": int(
                        ssl_test_metrics.get("support_non_cancer", 0)
                    ),
                    "test_precision_cancer": float(
                        ssl_test_metrics.get("precision_cancer", float("nan"))
                    ),
                    "test_recall_cancer": float(
                        ssl_test_metrics.get("recall_cancer", float("nan"))
                    ),
                    "test_f1_cancer": float(
                        ssl_test_metrics.get("f1_cancer", float("nan"))
                    ),
                    "test_support_cancer": int(ssl_test_metrics.get("support_cancer", 0)),
                }
            )
            run_agg_path = _write_aggregated_results(run_dir, aggregated_rows)
            test_modes_path = _write_test_metrics_by_mode(run_dir, test_mode_rows)
            global_agg_path = _append_global_aggregated_results(
                cfg.output_dir, [aggregated_rows[-1]]
            )
            print(f"[AGG] updated run aggregate: {run_agg_path}")
            print(f"[AGG] updated global aggregate: {global_agg_path}")
            print(f"[AGG] updated per-mode test metrics: {test_modes_path}")
        run_random = True
        if run_random:
            random_metrics_path, _, random_summary, random_test_metrics = _run_experiment(
                run_name="random",
                cfg=cfg,
                run_dir=phase_dir,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_weights=class_weights,
                encoder_checkpoint=None,
            )
            aggregated_rows.append(
                {
                    "run_id": run_id,
                    "phase": phase_name,
                    "few_shot_pct": few_shot_pct,
                    "mode": "random",
                    "train_samples": len(train_samples),
                    "train_normal": train_counts.get(0, 0),
                    "train_cancer": train_counts.get(1, 0),
                    **random_summary,
                }
            )
            test_mode_rows.append(
                {
                    "run_id": run_id,
                    "phase": phase_name,
                    "few_shot_pct": few_shot_pct,
                    "mode": "random",
                    "test_loss": float(random_test_metrics.get("loss", float("nan"))),
                    "test_acc": float(random_test_metrics.get("acc", float("nan"))),
                    "test_balanced_acc": float(
                        random_test_metrics.get("balanced_acc", float("nan"))
                    ),
                    "test_precision_non_cancer": float(
                        random_test_metrics.get("precision_non_cancer", float("nan"))
                    ),
                    "test_recall_non_cancer": float(
                        random_test_metrics.get("recall_non_cancer", float("nan"))
                    ),
                    "test_f1_non_cancer": float(
                        random_test_metrics.get("f1_non_cancer", float("nan"))
                    ),
                    "test_support_non_cancer": int(
                        random_test_metrics.get("support_non_cancer", 0)
                    ),
                    "test_precision_cancer": float(
                        random_test_metrics.get("precision_cancer", float("nan"))
                    ),
                    "test_recall_cancer": float(
                        random_test_metrics.get("recall_cancer", float("nan"))
                    ),
                    "test_f1_cancer": float(
                        random_test_metrics.get("f1_cancer", float("nan"))
                    ),
                    "test_support_cancer": int(random_test_metrics.get("support_cancer", 0)),
                }
            )
            run_agg_path = _write_aggregated_results(run_dir, aggregated_rows)
            test_modes_path = _write_test_metrics_by_mode(run_dir, test_mode_rows)
            global_agg_path = _append_global_aggregated_results(
                cfg.output_dir, [aggregated_rows[-1]]
            )
            print(f"[AGG] updated run aggregate: {run_agg_path}")
            print(f"[AGG] updated global aggregate: {global_agg_path}")
            print(f"[AGG] updated per-mode test metrics: {test_modes_path}")

    if aggregated_rows:
        print(f"[AGG] aggregation complete for run: {run_dir}")
    return run_dir


def _run_seed_sweep(
    cfg: TrainConfig,
    args: argparse.Namespace,
    *,
    output_dir: Path,
    run_name_prefix: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seeds = _resolve_split_seeds(cfg)
    for seed in seeds:
        seed_cfg = deepcopy(cfg)
        seed_cfg.output_dir = output_dir
        seed_cfg.split_seed = int(seed)
        seed_cfg.split_seeds = None

        run_id = f"{run_name_prefix}__seed_{seed}"
        print(f"[SEED] running seed={seed} run_id={run_id}")
        run_dir = _run_single_config(seed_cfg, args, run_id_override=run_id)
        rows.append(
            {
                "seed": seed,
                "run_id": run_id,
                "run_dir": str(run_dir),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    cfg = _build_config(args)
    checkpoint = Path(cfg.encoder_checkpoint) if cfg.encoder_checkpoint else None
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(
            "MuellerPT checkpoint not found: "
            f"{checkpoint}. Set MUELLERPT_CHECKPOINT or pass "
            "--encoder-checkpoint."
        )
    seeds = _resolve_split_seeds(cfg)
    if len(seeds) != 30:
        raise ValueError(f"The publication table requires exactly 30 split seeds; got {len(seeds)}.")

    seed_sweep_id = cfg.run_name or "paper_30_seed_sweep"
    seed_sweep_dir = cfg.output_dir / seed_sweep_id
    seed_sweep_dir.mkdir(parents=True, exist_ok=True)
    meta = {"name": "paper_30_seed_sweep", "seeds": seeds}
    (seed_sweep_dir / "seed_sweep_used.json").write_text(json.dumps(meta, indent=2))
    rows = _run_seed_sweep(
        cfg,
        args,
        output_dir=seed_sweep_dir,
        run_name_prefix=seed_sweep_id,
    )
    progress_path = _write_sweep_progress(seed_sweep_dir, rows)
    print(f"[SEED] complete: {seed_sweep_dir}")
    print(f"[SEED] progress updated: {progress_path}")


if __name__ == "__main__":
    main()
