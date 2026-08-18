# PoLambRimetry tissue segmentation

This experiment compares pretrained and scratch HRNet encoders for grey/white-matter segmentation using six-specimen nested cross-validation and label fractions of 1%, 5%, 25%, 50%, and 100%.

This directory implements only the paper configuration: raw 16-channel Mueller input (the former Option A), M00 fusion, seed 400, deterministic execution, and no batch-normalization reset. The unused decomposition-input and M00-only ablations have been removed.

## Inputs

Download PoLambRimetry from the
[University of Cantabria dataset record](https://web.unican.es/portal-investigador/en/datasets/dataset-detail?ds=DTS23011).
Set `MUELLERPT_POLAMBRIMETRY_ROOT` to the extracted dataset and
`MUELLERPT_CHECKPOINT` to the released epoch-150 checkpoint.

## Paper experiment

From the repository root:

```bash
PYTHON_BIN=.venv/bin/python scripts/reproduce_polambrimetry_table.sh
```

Outputs are written beneath:

```text
$MUELLERPT_OUTPUT_ROOT/results/polambrimetry/paper_nested_cv/
```

The launcher uses `--resume`, allowing completed nested-CV entries to be retained after interruption.

The deterministic controls are enabled where PyTorch supports them. Some CUDA
operations only support warning mode, so freshly trained metrics can still vary
with hardware and software versions and are not expected to match the reference
values exactly.

For each of six held-out test specimens, each of the other five specimens is used once for validation. The reported mean and sample standard deviation therefore use all 30 nested-CV test/validation pairs with equal weight.

## Summaries

The launcher trains and evaluates the nested folds; run the table summarizer
separately after training finishes:

```bash
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
"$PYTHON_BIN" experiments/polambrimetry/summarize_metrics.py
```

Sanitized overall and GM/WM reference tables are available in `results/reference_tables/`.
