#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
import os
from pathlib import Path



# -----------------------------
# User-editable paths/settings.
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("MUELLERPT_OUTPUT_ROOT", str(REPO_ROOT / "outputs")))
INPUT_CSV_PATH = OUTPUT_ROOT / "results" / "polambrimetry" / "paper_nested_cv" / "nested_summary.csv"
OUTPUT_DIR = OUTPUT_ROOT / "results" / "polambrimetry" / "paper_nested_cv" / "summary"

RUN_COLUMN = "run"
FEW_SHOT_COLUMN = "few_shot_pct"
DICE_COLUMN = "test_mean_dice"

RANDOM_RUN_NAMES = {"random"}
PRETRAINED_RUN_NAMES = {"ssl_pretrained", "pretrained"}

# Keep full-data runs (`few_shot_pct == 100`) in outputs when present.
INCLUDE_100_PERCENT = True


def _parse_float(text: str) -> float | None:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def _parse_int(text: str) -> int | None:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def _canonical_encoder(run_name: str) -> str | None:
    name = str(run_name).strip().lower()
    if name in RANDOM_RUN_NAMES:
        return "random"
    if name in PRETRAINED_RUN_NAMES:
        return "pretrained"
    return None


def _extract_class_columns(header: list[str]) -> list[tuple[str, str]]:
    class_cols: list[tuple[str, str]] = []
    prefix = "test_dice_"
    for col in header:
        if not col.startswith(prefix):
            continue
        class_name = col[len(prefix) :].strip()
        if not class_name:
            continue
        class_cols.append((col, class_name))
    return class_cols


def _collect_rows(
    input_csv: Path,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[str],
]:
    grouped: dict[tuple[str, int], list[float]] = {}
    grouped_per_class: dict[tuple[str, str, int], list[float]] = {}
    raw_rows: list[dict[str, str]] = []
    raw_per_class_rows: list[dict[str, str]] = []
    skipped_runs: set[str] = set()

    with input_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        header = set(fieldnames)
        required = {RUN_COLUMN, FEW_SHOT_COLUMN, DICE_COLUMN}
        missing = sorted(required - header)
        if missing:
            raise ValueError(f"Missing required columns in {input_csv}: {missing}")
        class_columns = _extract_class_columns(fieldnames)
        for row in reader:
            encoder = _canonical_encoder(row.get(RUN_COLUMN, ""))
            if encoder is None:
                skipped_runs.add(str(row.get(RUN_COLUMN, "")))
                continue

            pct = _parse_int(row.get(FEW_SHOT_COLUMN, ""))
            if pct is None:
                continue
            if not INCLUDE_100_PERCENT and pct == 100:
                continue

            mean_dice = _parse_float(row.get(DICE_COLUMN, ""))
            if mean_dice is not None:
                key = (encoder, pct)
                grouped.setdefault(key, []).append(mean_dice)
                raw_rows.append(
                    {
                        "encoder": encoder,
                        "few_shot_pct": str(pct),
                        "test_mean_dice": f"{mean_dice:.6f}",
                        "pair_index": str(row.get("pair_index", "")),
                        "outer_fold": str(row.get("outer_fold", "")),
                        "test_specimen": str(row.get("test_specimen", "")),
                        "val_specimen": str(row.get("val_specimen", "")),
                    }
                )

            for col, class_name in class_columns:
                class_dice = _parse_float(row.get(col, ""))
                if class_dice is None:
                    continue
                key_class = (class_name, encoder, pct)
                grouped_per_class.setdefault(key_class, []).append(class_dice)
                raw_per_class_rows.append(
                    {
                        "class_name": class_name,
                        "encoder": encoder,
                        "few_shot_pct": str(pct),
                        "test_dice": f"{class_dice:.6f}",
                        "pair_index": str(row.get("pair_index", "")),
                        "outer_fold": str(row.get("outer_fold", "")),
                        "test_specimen": str(row.get("test_specimen", "")),
                        "val_specimen": str(row.get("val_specimen", "")),
                    }
                )

    aggregate_rows: list[dict[str, str]] = []
    for (encoder, pct), values in grouped.items():
        n = len(values)
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 0 else float("nan")
        ci95 = 1.96 * sem if n > 1 else 0.0
        aggregate_rows.append(
            {
                "few_shot_pct": str(pct),
                "encoder": encoder,
                "n": str(n),
                "mean_dice": f"{mean:.6f}",
                "std_dice": f"{std:.6f}",
                "sem_dice": f"{sem:.6f}",
                "ci95_dice": f"{ci95:.6f}",
            }
        )

    per_class_aggregate_rows: list[dict[str, str]] = []
    for (class_name, encoder, pct), values in grouped_per_class.items():
        n = len(values)
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 0 else float("nan")
        ci95 = 1.96 * sem if n > 1 else 0.0
        per_class_aggregate_rows.append(
            {
                "few_shot_pct": str(pct),
                "class_name": class_name,
                "encoder": encoder,
                "n": str(n),
                "mean_dice": f"{mean:.6f}",
                "std_dice": f"{std:.6f}",
                "sem_dice": f"{sem:.6f}",
                "ci95_dice": f"{ci95:.6f}",
            }
        )

    aggregate_rows.sort(key=lambda r: (int(r["few_shot_pct"]), r["encoder"]))
    per_class_aggregate_rows.sort(
        key=lambda r: (int(r["few_shot_pct"]), r["class_name"], r["encoder"])
    )
    skipped = sorted(x for x in skipped_runs if x)
    return (
        raw_rows,
        aggregate_rows,
        raw_per_class_rows,
        per_class_aggregate_rows,
        skipped,
    )


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_wide_rows(aggregate_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_pct: dict[int, dict[str, dict[str, str]]] = {}
    for row in aggregate_rows:
        pct = int(row["few_shot_pct"])
        by_pct.setdefault(pct, {})[row["encoder"]] = row

    wide_rows: list[dict[str, str]] = []
    for pct in sorted(by_pct):
        pre = by_pct[pct].get("pretrained")
        rnd = by_pct[pct].get("random")
        wide_rows.append(
            {
                "few_shot_pct": str(pct),
                "pretrained_mean_dice": pre["mean_dice"] if pre else "",
                "pretrained_std_dice": pre["std_dice"] if pre else "",
                "pretrained_n": pre["n"] if pre else "",
                "random_mean_dice": rnd["mean_dice"] if rnd else "",
                "random_std_dice": rnd["std_dice"] if rnd else "",
                "random_n": rnd["n"] if rnd else "",
            }
        )
    return wide_rows


def _build_per_class_wide_rows(
    per_class_aggregate_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_key: dict[tuple[int, str], dict[str, dict[str, str]]] = {}
    for row in per_class_aggregate_rows:
        pct = int(row["few_shot_pct"])
        class_name = row["class_name"]
        by_key.setdefault((pct, class_name), {})[row["encoder"]] = row

    wide_rows: list[dict[str, str]] = []
    for pct, class_name in sorted(by_key.keys(), key=lambda x: (x[0], x[1])):
        pre = by_key[(pct, class_name)].get("pretrained")
        rnd = by_key[(pct, class_name)].get("random")
        wide_rows.append(
            {
                "few_shot_pct": str(pct),
                "class_name": class_name,
                "pretrained_mean_dice": pre["mean_dice"] if pre else "",
                "pretrained_std_dice": pre["std_dice"] if pre else "",
                "pretrained_n": pre["n"] if pre else "",
                "random_mean_dice": rnd["mean_dice"] if rnd else "",
                "random_std_dice": rnd["std_dice"] if rnd else "",
                "random_n": rnd["n"] if rnd else "",
            }
        )
    return wide_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize few-shot PoLambRimetry Dice metrics.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    input_csv = args.input_csv.expanduser()
    outdir = args.output_dir.expanduser()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    (
        raw_rows,
        aggregate_rows,
        raw_per_class_rows,
        per_class_aggregate_rows,
        skipped,
    ) = _collect_rows(input_csv)
    if not aggregate_rows:
        raise ValueError(
            "No aggregated rows were produced. Check run names and input columns."
        )

    raw_csv = outdir / "few_shot_dice_raw_rows.csv"
    long_csv = outdir / "few_shot_dice_summary_long.csv"
    wide_csv = outdir / "few_shot_dice_summary_wide.csv"
    raw_per_class_csv = outdir / "few_shot_dice_per_class_raw_rows.csv"
    long_per_class_csv = outdir / "few_shot_dice_per_class_summary_long.csv"
    wide_per_class_csv = outdir / "few_shot_dice_per_class_summary_wide.csv"

    _write_csv(
        raw_csv,
        raw_rows,
        [
            "encoder",
            "few_shot_pct",
            "test_mean_dice",
            "pair_index",
            "outer_fold",
            "test_specimen",
            "val_specimen",
        ],
    )
    _write_csv(
        long_csv,
        aggregate_rows,
        ["few_shot_pct", "encoder", "n", "mean_dice", "std_dice", "sem_dice", "ci95_dice"],
    )
    _write_csv(
        wide_csv,
        _build_wide_rows(aggregate_rows),
        [
            "few_shot_pct",
            "pretrained_mean_dice",
            "pretrained_std_dice",
            "pretrained_n",
            "random_mean_dice",
            "random_std_dice",
            "random_n",
        ],
    )
    if raw_per_class_rows:
        _write_csv(
            raw_per_class_csv,
            raw_per_class_rows,
            [
                "class_name",
                "encoder",
                "few_shot_pct",
                "test_dice",
                "pair_index",
                "outer_fold",
                "test_specimen",
                "val_specimen",
            ],
        )
    if per_class_aggregate_rows:
        _write_csv(
            long_per_class_csv,
            per_class_aggregate_rows,
            [
                "few_shot_pct",
                "class_name",
                "encoder",
                "n",
                "mean_dice",
                "std_dice",
                "sem_dice",
                "ci95_dice",
            ],
        )
        _write_csv(
            wide_per_class_csv,
            _build_per_class_wide_rows(per_class_aggregate_rows),
            [
                "few_shot_pct",
                "class_name",
                "pretrained_mean_dice",
                "pretrained_std_dice",
                "pretrained_n",
                "random_mean_dice",
                "random_std_dice",
                "random_n",
            ],
        )
    print(f"[DONE] Wrote: {raw_csv}")
    print(f"[DONE] Wrote: {long_csv}")
    print(f"[DONE] Wrote: {wide_csv}")
    if raw_per_class_rows:
        print(f"[DONE] Wrote: {raw_per_class_csv}")
    if per_class_aggregate_rows:
        print(f"[DONE] Wrote: {long_per_class_csv}")
        print(f"[DONE] Wrote: {wide_per_class_csv}")
    if skipped:
        print(f"[INFO] Ignored run names not in mapping: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
