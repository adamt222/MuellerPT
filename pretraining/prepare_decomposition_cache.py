#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PRETRAINING_ROOT = Path(__file__).resolve().parent
if str(PRETRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAINING_ROOT))

from config_pretrain import DATASET_NAME, DECOMP_CACHE_SUFFIX, FILTERED_MATRICES_ROOT
from utils import compute_decomp_maps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the cached Lu-Chipman targets used by MuellerPT."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=FILTERED_MATRICES_ROOT / DATASET_NAME,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = sorted(args.dataset_dir.rglob("*_filtered.npy"))
    if not inputs:
        raise FileNotFoundError(
            f"No *_filtered.npy inputs found beneath {args.dataset_dir}."
        )

    written = 0
    for index, input_path in enumerate(inputs, start=1):
        output_path = input_path.with_name(
            f"{input_path.stem}{DECOMP_CACHE_SUFFIX}"
        )
        if output_path.exists() and not args.overwrite:
            continue
        mueller = np.load(input_path, allow_pickle=False)
        targets, roi_mask = compute_decomp_maps(mueller)
        np.savez_compressed(output_path, **targets, roi_mask=roi_mask)
        written += 1
        print(f"[{index}/{len(inputs)}] {output_path}")

    print(f"[DONE] Wrote {written} decomposition caches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
