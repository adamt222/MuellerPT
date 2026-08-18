#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${MUELLERPT_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" "$REPO_ROOT/experiments/colopola/train_unet.py" \
  --output-dir "$OUTPUT_ROOT/results/colopola" \
  --run-name paper_30_seed_sweep \
  --split-seeds \
    42 400 4000 5000 \
    6000 7000 8000 9000 10000 \
    10001 10002 10003 10004 10005 10006 10007 10008 10009 10010 \
    20020 30030 40040 50050 60060 70070 80080 90090 101010 \
    101011 101012 \
  --few-shot-percentages 1 5 25 50 100
