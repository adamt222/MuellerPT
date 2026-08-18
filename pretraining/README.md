# MuellerPT pretraining

This directory reproduces the single pretraining protocol used to create the
released epoch-150 HRNet-W18 checkpoint. Exploratory architectures,
normalization modes, validation splits, patch sampling, alternate channel
corruption policies, and diagnostic plotting are intentionally omitted.

## Required inputs

Download MAP-Org from its
[Zenodo record](https://doi.org/10.5281/zenodo.20274683). The code expects the
prepared, physically filtered arrays described below.

Place the `0902Measurement` files beneath
`$MUELLERPT_INPUT_ROOT/filtered_matrices/0902Measurement`. Each Mueller input
must have two sidecars in the same directory:

```text
<name>_filtered.npy
<name>_filtered_m00.npy
<name>_filtered_decomp.npz
```

The decomposition cache must contain `retardance`, `depolarization`, and
`diattenuation`. If it is not supplied with the data, generate it with:

```bash
"${PYTHON_BIN:-python3}" pretraining/prepare_decomposition_cache.py
```

## Train

```bash
export MUELLERPT_INPUT_ROOT=/path/to/prepared/data
export MUELLERPT_OUTPUT_ROOT=/path/on/mounted/drive/MuellerPT
export PYTHON_BIN="$PWD/.venv/bin/python"

"$PYTHON_BIN" pretraining/train_pretrain.py
```

The fixed protocol uses 600×600 center crops, expand-and-crop Mueller-aware
rotation, the three published channel-drop presets, M00 fusion and dropout,
cached Lu–Chipman targets, and the decomposition smooth-L1 plus gradient loss.
It trains for 150 epochs and saves a checkpoint every 10 epochs beneath
`$MUELLERPT_OUTPUT_ROOT/results/pretraining`.

Portable paths and the run name are defined in `config_pretrain.py`; all
method-defining values match the released checkpoint configuration.

See the main [reproducibility notes](../README.md#reproducibility-expectations)
for the expected variation between training runs.
