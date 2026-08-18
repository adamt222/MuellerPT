#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${MUELLERPT_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" "$REPO_ROOT/experiments/polambrimetry/train_unet.py" \
  --outdir "$OUTPUT_ROOT/results/polambrimetry" \
  --run-id paper_nested_cv \
  --resume \
  --few-shot-percentages 1 5 25 50 100
