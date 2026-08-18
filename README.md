# MuellerPT

MuellerPT provides self-supervised HRNet pretraining for Mueller-matrix images and two downstream paper experiments: cancer classification on ColoPola and grey/white-matter segmentation on PoLambRimetry.

The code is intentionally publication-focused. Pretraining contains only the protocol used for the released epoch-150 checkpoint, and downstream experiments contain only the configurations used for the reported tables.

## Repository layout

```text
pretraining/                 Shared MuellerPT encoder and pretraining code
experiments/colopola/        Cancer classification experiment
experiments/polambrimetry/   Grey/white-matter segmentation experiment
scripts/                     Exact paper reproduction launchers
results/reference_tables/    Sanitized aggregate paper results
tests/                       Portable repository tests
```

Generated data, checkpoints, logs, and per-run outputs are deliberately excluded from Git.

## Installation

Python 3.12 was used for validation.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For development and tests:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

The last local validation used Ubuntu Linux, Python 3.12.3, PyTorch
2.10.0+cu128, torchvision 0.25.0+cu128, CUDA 12.8, cuDNN 9.1, timm 1.0.24,
NumPy 1.26.4, h5py 3.15.1, and an NVIDIA RTX 4070 Ti SUPER. The pinned
requirements are the supported reference environment; other compatible CUDA
and CPU builds may also work.

## Data and output locations

Portable defaults are relative to the repository:

```text
data/filtered_matrices/       Filtered Mueller-matrix inputs
data/polambrimetry/           PoLambRimetry inputs
outputs/checkpoints/          Downloaded model weights
outputs/results/              Generated experiment results
```

Override them without editing code:

```bash
export MUELLERPT_INPUT_ROOT=/path/to/data
export MUELLERPT_POLAMBRIMETRY_ROOT=/path/to/polambrimetry
export MUELLERPT_OUTPUT_ROOT=/path/to/outputs
export MUELLERPT_CHECKPOINT=/path/to/muellerpt_hrnet_w18_epoch150.pt
```

The downstream experiments use `muellerpt_hrnet_w18_epoch150.pt`. Its expected SHA-256 is:

```text
ea85d10e1340cb2be7577783a460fdbbb0312063e05392d44006f0a31d9d6d42
```

The datasets and checkpoint are not redistributed in this repository. Obtain
them according to their respective access conditions and place them at the
paths above. The training code consumes the filtered layouts described in the
experiment READMEs; the public dataset downloads may require preparation into
those layouts.

Dataset sources:

- [ColoPola (Zenodo, DOI 10.5281/zenodo.10554304)](https://doi.org/10.5281/zenodo.10554304)
- [PoLambRimetry (University of Cantabria dataset record)](https://web.unican.es/portal-investigador/en/datasets/dataset-detail?ds=DTS23011)
- [MAP-Org (Zenodo, DOI 10.5281/zenodo.20274683)](https://doi.org/10.5281/zenodo.20274683)

## Reproduce the paper experiments

Run from the repository root. This example keeps generated results on the
mounted data drive:

```bash
export MUELLERPT_INPUT_ROOT=/path/to/prepared/data
export MUELLERPT_POLAMBRIMETRY_ROOT=/path/to/PoLambRimetry
export MUELLERPT_OUTPUT_ROOT=/path/on/mounted/drive/MuellerPT
export MUELLERPT_CHECKPOINT="$MUELLERPT_OUTPUT_ROOT/checkpoints/muellerpt_hrnet_w18_epoch150.pt"
export PYTHON_BIN="$PWD/.venv/bin/python"

scripts/reproduce_colopola_table.sh
scripts/reproduce_polambrimetry_table.sh
```

The ColoPola launcher performs 30 seeded runs at five label percentages for both pretrained and scratch models. The PoLambRimetry launcher performs the six-specimen nested cross-validation experiment and supports resume.

The launchers run model training and evaluation. They do not automatically run
the separate table-aggregation commands below.

Aggregate the generated results:

```bash
"$PYTHON_BIN" experiments/colopola/aggregate_seed_results.py
"$PYTHON_BIN" experiments/polambrimetry/summarize_metrics.py
```

Reference aggregate tables are provided in [`results/reference_tables`](results/reference_tables). They contain no local paths, raw predictions, seed-level records, or specimen identifiers.

## Reproducibility expectations

The launchers reproduce the published protocols, not guaranteed bit-for-bit
metric values. Training remains stochastic because of initialization, data
sampling and augmentation, and hardware- or library-dependent GPU kernels.
PoLambRimetry enables deterministic settings where supported, but PyTorch may
warn and continue when a CUDA operation has no deterministic implementation.
Fresh means and standard deviations are therefore not expected to match the
reference tables exactly; compare the aggregate trend and sampling variation,
and record the software and hardware environment used.

## Pretraining

From the repository root:

```bash
export MUELLERPT_INPUT_ROOT=/path/to/prepared/data
export MUELLERPT_OUTPUT_ROOT=/path/on/mounted/drive/
export PYTHON_BIN="$PWD/.venv/bin/python"

# Run this once only if *_filtered_decomp.npz sidecars are absent.
"$PYTHON_BIN" pretraining/prepare_decomposition_cache.py

# Train the published 150-epoch protocol.
"$PYTHON_BIN" pretraining/train_pretrain.py
```

The cache command is only needed when the Lu–Chipman `_decomp.npz` sidecars are
not already present. See the pretraining and experiment-specific READMEs for
input formats and configuration details.

## Citation

Please cite the software and associated paper using [`CITATION.cff`](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).
