from __future__ import annotations

import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path(os.environ.get("MUELLERPT_INPUT_ROOT", str(REPO_ROOT / "data")))
OUTPUT_ROOT = Path(os.environ.get("MUELLERPT_OUTPUT_ROOT", str(REPO_ROOT / "outputs")))

# Exact configuration used by the published epoch-150 checkpoint.
DATASET_NAME = "0902Measurement"
FILTERED_MATRICES_ROOT = INPUT_ROOT / "filtered_matrices"
OUTPUT_DIR = OUTPUT_ROOT / "results" / "pretraining"
RUN_NAME: str | None = None

CENTER_CROP_SIZE = (600, 600)
SEED = 0
BATCH_SIZE = 12
EPOCHS = 150
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 1e-3
WEIGHT_DECAY = 1e-3
GRAD_CLIP_MAX_NORM = 1.0
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 5
LR_PLATEAU_MIN_LR = 1e-6
SAVE_EVERY_N_EPOCHS = 10

DECOMP_ORDER = ("retardance", "depolarization", "diattenuation")
DECOMP_CACHE_SUFFIX = "_decomp.npz"

CHANNEL_DROP_PRESETS = (
    "upper_left_3x3",
    "drop_last_column",
    "drop_last_row",
)
CHANNEL_DROP_KEEP_FULL_PROB = 0.5

M00_ROOT = FILTERED_MATRICES_ROOT
M00_SUFFIX = "_m00.npy"
M00_DROP_PROB = 0.3

ROTATION_DEGREES = (-60, 60)
ROTATION_P = 0.8
