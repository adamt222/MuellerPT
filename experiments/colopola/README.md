# ColoPola cancer classification

This experiment compares an HRNet encoder initialized from MuellerPT pretraining with the same architecture trained from scratch. It reports accuracy, cancer sensitivity, and non-cancer specificity at 1%, 5%, 25%, 50%, and 100% of labelled training data.

Only the publication workflow is included: all 16 Mueller channels, the fixed 60/20/20 stratified split protocol, the exact 30 seeds, and both pretrained and scratch models. Hyperparameter-grid sweeps, channel ablations, encoder freezing, and batch-normalization re-estimation have been removed.

## Inputs

Download ColoPola from its
[Zenodo record](https://doi.org/10.5281/zenodo.10554304). This experiment reads
the prepared, physically filtered arrays rather than the archive's raw layout.
Filtered samples are expected beneath:

```text
$MUELLERPT_INPUT_ROOT/filtered_matrices/colopola/
```

Each filename must begin with `normal__` or `cancer__` and end in `_filtered.npy`. Set `MUELLERPT_CHECKPOINT` to the released epoch-150 checkpoint.

## Paper experiment

From the repository root:

```bash
PYTHON_BIN=.venv/bin/python scripts/reproduce_colopola_table.sh
```

The launcher contains the exact 30 publication seeds and writes to:

```text
$MUELLERPT_OUTPUT_ROOT/results/colopola/paper_30_seed_sweep/
```

The original experiment used `deterministic=False`. See the main
[reproducibility notes](../../README.md#reproducibility-expectations) for the
expected variation between runs.

Each seed produces one complete-test-set Accuracy, Sensitivity, and Specificity value. The table reports the equally weighted mean and sample standard deviation of those 30 seed-level values.

## Aggregate results

The launcher above trains and evaluates the models; run aggregation separately
after all 30 seeds finish:

```bash
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
"$PYTHON_BIN" experiments/colopola/aggregate_seed_results.py
```

The sanitized reference table is `results/reference_tables/colopola_classification.csv`.
