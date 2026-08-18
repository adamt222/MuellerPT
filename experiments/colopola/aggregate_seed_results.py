#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("MUELLERPT_OUTPUT_ROOT", str(REPO_ROOT / "outputs")))


MODE_LABELS = {
    "ssl_pretrained": "Pre-trained",
    "random": "Random init",
}
MODE_ORDER = ["ssl_pretrained", "random"]
DEFAULT_PUBLICATION_METRICS = [
    "test_acc",
    "test_recall",
    "test_specificity",
]

SEED_SUFFIX_RE = re.compile(r"__seed_(?P<seed>\d+)$")
FEW_SHOT_PHASE_RE = re.compile(r"few_shot_(?P<pct>\d+)")


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return float("nan")
    return sum(items) / len(items)


def _std(values: List[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _sem(values: List[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    return _std(values) / math.sqrt(len(values))


def _fmt_mean_std(mean_val: float, std_val: float, decimals: int) -> str:
    if math.isnan(mean_val):
        return ""
    if math.isnan(std_val):
        return f"{mean_val:.{decimals}f}"
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _extract_few_shot_pct(row: Dict[str, str]) -> Optional[int]:
    pct = _parse_int(row.get("few_shot_pct"))
    if pct is not None:
        return pct
    phase = str(row.get("phase", "")).strip()
    if phase == "full_data":
        return 100
    m = FEW_SHOT_PHASE_RE.search(phase)
    if m:
        try:
            return int(m.group("pct"))
        except ValueError:
            return None
    return None


def _infer_seed_from_run_dir(run_dir: Path) -> Optional[int]:
    m = SEED_SUFFIX_RE.search(run_dir.name)
    if not m:
        return None
    try:
        return int(m.group("seed"))
    except ValueError:
        return None


def _collect_seed_run_dirs(seed_sweep_dir: Path) -> List[Dict[str, object]]:
    sweep_csv = seed_sweep_dir / "sweep_trials.csv"
    rows: List[Dict[str, object]] = []
    if sweep_csv.exists():
        with sweep_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            has_seed_col = "seed" in (reader.fieldnames or [])
            for row in reader:
                run_dir_raw = str(row.get("run_dir", "")).strip()
                if not run_dir_raw:
                    continue
                run_dir = Path(run_dir_raw)
                if not run_dir.exists():
                    continue
                seed = _parse_int(row.get("seed")) if has_seed_col else None
                if seed is None:
                    seed = _infer_seed_from_run_dir(run_dir)
                rows.append(
                    {
                        "seed": seed,
                        "run_dir": run_dir,
                        "seed_sweep_dir": seed_sweep_dir.resolve(),
                    }
                )
    if rows:
        # Remove duplicates by resolved path.
        seen: set[Path] = set()
        deduped: List[Dict[str, object]] = []
        for item in rows:
            run_dir = Path(item["run_dir"]).resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            deduped.append(
                {
                    "seed": item["seed"],
                    "run_dir": run_dir,
                    "seed_sweep_dir": Path(item["seed_sweep_dir"]).resolve(),
                }
            )
        return deduped

    # Fallback: scan subfolders named *__seed_<N>.
    found: List[Dict[str, object]] = []
    for sub in sorted(seed_sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        seed = _infer_seed_from_run_dir(sub)
        if seed is None:
            continue
        if (sub / "aggregated_results.csv").exists() or (sub / "test_metrics_by_mode.csv").exists():
            found.append(
                {
                    "seed": seed,
                    "run_dir": sub.resolve(),
                    "seed_sweep_dir": seed_sweep_dir.resolve(),
                }
            )
    if not found:
        raise FileNotFoundError(
            f"No seed run directories found in {seed_sweep_dir}."
        )
    return found


def _read_test_metrics_by_mode(path: Path) -> Dict[tuple[int, str, str], Dict[str, float]]:
    out: Dict[tuple[int, str, str], Dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            few_shot = _extract_few_shot_pct(row)
            mode = str(row.get("mode", "")).strip()
            phase = str(row.get("phase", "")).strip()
            if few_shot is None or not mode:
                continue
            key = (few_shot, mode, phase)
            metrics: Dict[str, float] = {}
            for k, v in row.items():
                if not k.startswith("test_"):
                    continue
                val = _parse_float(v)
                if val is not None:
                    metrics[k] = val
            out[key] = metrics
    return out


def _read_seed_records(seed_runs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for item in seed_runs:
        seed = item.get("seed")
        run_dir = Path(item["run_dir"])
        seed_sweep_dir = Path(item.get("seed_sweep_dir", run_dir.parent))
        agg_path = run_dir / "aggregated_results.csv"
        if not agg_path.exists():
            print(f"[WARN] Missing aggregated_results.csv: {agg_path}")
            continue
        per_class_map = _read_test_metrics_by_mode(run_dir / "test_metrics_by_mode.csv")
        with agg_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                few_shot = _extract_few_shot_pct(row)
                phase = str(row.get("phase", "")).strip()
                mode = str(row.get("mode", "")).strip()
                if few_shot is None or not mode:
                    continue
                rec: Dict[str, object] = {
                    "seed": seed if seed is not None else _parse_int(row.get("split_seed")),
                    "run_id": str(row.get("run_id", "")).strip() or run_dir.name,
                    "run_dir": str(run_dir),
                    "seed_sweep_id": seed_sweep_dir.name,
                    "seed_sweep_dir": str(seed_sweep_dir),
                    "phase": phase,
                    "few_shot_pct": few_shot,
                    "mode": mode,
                    "mode_label": MODE_LABELS.get(mode, mode),
                }
                for k, v in row.items():
                    if not k.startswith("test_"):
                        continue
                    val = _parse_float(v)
                    if val is not None:
                        rec[k] = val
                extra = per_class_map.get((few_shot, mode, phase), {})
                for k, v in extra.items():
                    rec[k] = v
                records.append(rec)
    return records


def _collect_metric_names(records: List[Dict[str, object]]) -> List[str]:
    metrics = set()
    for rec in records:
        for key, value in rec.items():
            if key.startswith("test_") and isinstance(value, (int, float)):
                metrics.add(key)
    preferred = [m for m in DEFAULT_PUBLICATION_METRICS if m in metrics]
    remaining = sorted(metrics.difference(preferred))
    return preferred + remaining


def _collect_publication_metrics(all_metrics: List[str]) -> List[str]:
    preferred = [m for m in DEFAULT_PUBLICATION_METRICS if m in all_metrics]
    if preferred:
        return preferred
    # Fallback to key test metrics if defaults are absent.
    fallback = [m for m in all_metrics if m in {"test_acc", "test_balanced_acc", "test_f1"}]
    if fallback:
        return fallback
    return list(all_metrics)


def _aggregate_records(
    records: List[Dict[str, object]],
    metrics: List[str],
) -> List[Dict[str, object]]:
    groups: Dict[tuple[int, str], List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        few_shot = int(rec["few_shot_pct"])
        mode = str(rec["mode"])
        groups[(few_shot, mode)].append(rec)

    out_rows: List[Dict[str, object]] = []
    for (few_shot, mode) in sorted(groups.keys(), key=lambda x: (x[0], MODE_ORDER.index(x[1]) if x[1] in MODE_ORDER else 99, x[1])):
        items = groups[(few_shot, mode)]
        seeds = sorted({int(r["seed"]) for r in items if r.get("seed") is not None})
        row: Dict[str, object] = {
            "few_shot_pct": few_shot,
            "mode": mode,
            "mode_label": MODE_LABELS.get(mode, mode),
            "n_runs": len(items),
            "n_seeds": len(seeds),
            "seeds": "|".join(str(s) for s in seeds),
        }
        for metric in metrics:
            vals: List[float] = []
            for rec in items:
                val = rec.get(metric)
                if isinstance(val, (int, float)) and not math.isnan(float(val)):
                    vals.append(float(val))
            if not vals:
                row[f"{metric}_mean"] = ""
                row[f"{metric}_std"] = ""
                row[f"{metric}_sem"] = ""
                row[f"{metric}_mean_std"] = ""
                continue
            m = _mean(vals)
            s = _std(vals)
            se = _sem(vals)
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
            row[f"{metric}_sem"] = se
            row[f"{metric}_mean_std"] = _fmt_mean_std(m, s, decimals=4)
        out_rows.append(row)
    return out_rows


def _build_publication_wide_table(
    summary_rows: List[Dict[str, object]],
    metrics: List[str],
    decimals: int,
) -> List[Dict[str, object]]:
    by_key: Dict[tuple[int, str], Dict[str, object]] = {}
    few_shots = sorted({int(row["few_shot_pct"]) for row in summary_rows})
    for row in summary_rows:
        by_key[(int(row["few_shot_pct"]), str(row["mode"]))] = row

    out_rows: List[Dict[str, object]] = []
    for few_shot in few_shots:
        out: Dict[str, object] = {"few_shot_pct": few_shot}
        n_by_mode = {}
        for mode in MODE_ORDER:
            row = by_key.get((few_shot, mode))
            n_by_mode[mode] = int(row["n_seeds"]) if row and row.get("n_seeds") not in ("", None) else 0
        out["n_seeds_pretrained"] = n_by_mode.get("ssl_pretrained", 0)
        out["n_seeds_random"] = n_by_mode.get("random", 0)

        for metric in metrics:
            pre_row = by_key.get((few_shot, "ssl_pretrained"))
            rnd_row = by_key.get((few_shot, "random"))
            pre_mean = float(pre_row.get(f"{metric}_mean")) if pre_row and pre_row.get(f"{metric}_mean") != "" else float("nan")
            pre_std = float(pre_row.get(f"{metric}_std")) if pre_row and pre_row.get(f"{metric}_std") != "" else float("nan")
            rnd_mean = float(rnd_row.get(f"{metric}_mean")) if rnd_row and rnd_row.get(f"{metric}_mean") != "" else float("nan")
            rnd_std = float(rnd_row.get(f"{metric}_std")) if rnd_row and rnd_row.get(f"{metric}_std") != "" else float("nan")

            out[f"pretrained_{metric}"] = _fmt_mean_std(pre_mean, pre_std, decimals=decimals)
            out[f"random_{metric}"] = _fmt_mean_std(rnd_mean, rnd_std, decimals=decimals)
        out_rows.append(out)
    return out_rows


def _filter_to_modes(records: List[Dict[str, object]], modes: List[str]) -> List[Dict[str, object]]:
    keep = set(modes)
    return [r for r in records if str(r.get("mode", "")) in keep]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Colopola seed runs into publication-ready CSVs "
            "(random vs pre-trained)."
        )
    )
    parser.add_argument(
        "--seed-sweep-dir",
        type=Path,
        default=OUTPUT_ROOT / "results" / "colopola" / "paper_30_seed_sweep",
        help="The paper_30_seed_sweep output directory.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory; defaults to <seed-sweep-dir>/publication_tables.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places for publication formatted values.",
    )
    args = parser.parse_args()

    seed_sweep_dir = args.seed_sweep_dir.resolve()
    outdir = args.outdir.resolve() if args.outdir is not None else seed_sweep_dir / "publication_tables"
    outdir.mkdir(parents=True, exist_ok=True)

    seed_runs = _collect_seed_run_dirs(seed_sweep_dir)
    print(f"[INFO] Seed sweep: {seed_sweep_dir}")
    print(f"[INFO] Found {len(seed_runs)} unique seed run directories.")
    records = _read_seed_records(seed_runs)
    if not records:
        raise RuntimeError(
            "No aggregated rows were loaded from selected seed run directories."
        )

    records = _filter_to_modes(records, ["ssl_pretrained", "random"])
    if not records:
        raise RuntimeError("No pretrained or scratch records were found.")

    metrics_all = _collect_metric_names(records)
    publication_metrics = _collect_publication_metrics(metrics_all)
    summary_rows = _aggregate_records(records, metrics_all)
    pub_rows = _build_publication_wide_table(summary_rows, publication_metrics, decimals=args.decimals)

    seed_level_path = outdir / "seed_level_results.csv"
    summary_long_path = outdir / "publication_summary_long.csv"
    publication_table_path = outdir / "publication_table_mean_std.csv"

    # Seed-level rows.
    seed_fields = [
        "seed_sweep_id",
        "seed_sweep_dir",
        "seed",
        "run_id",
        "run_dir",
        "phase",
        "few_shot_pct",
        "mode",
        "mode_label",
    ] + metrics_all
    _write_csv(seed_level_path, records, seed_fields)

    # Long numeric summary by (few-shot, mode).
    long_fields = ["few_shot_pct", "mode", "mode_label", "n_runs", "n_seeds", "seeds"]
    for metric in metrics_all:
        long_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_sem",
                f"{metric}_mean_std",
            ]
        )
    _write_csv(summary_long_path, summary_rows, long_fields)

    # Wide publication-ready table (one row per few-shot).
    wide_fields = ["few_shot_pct", "n_seeds_pretrained", "n_seeds_random"]
    for metric in publication_metrics:
        wide_fields.extend(
            [
                f"pretrained_{metric}",
                f"random_{metric}",
            ]
        )
    _write_csv(publication_table_path, pub_rows, wide_fields)

    print(f"[DONE] Wrote seed-level rows: {seed_level_path}")
    print(f"[DONE] Wrote long summary:   {summary_long_path}")
    print(f"[DONE] Wrote publication:    {publication_table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
