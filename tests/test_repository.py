from __future__ import annotations

import ast
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (ROOT / "pretraining", ROOT / "experiments", ROOT / "tests")


def test_all_python_files_parse() -> None:
    for source_root in SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))


def test_public_files_have_no_private_or_legacy_paths() -> None:
    forbidden = (
        "/home/" + "adam",
        "/mnt/" + "data",
        "Documents/" + "miccai",
        "ssl_" + "variant3",
        "colopola_" + "test",
        "Lu_polamb_" + "test2",
        "lu_polamb_" + "test2",
        "metric_" + "summary",
    )
    extensions = {".py", ".sh", ".md", ".json", ".yml", ".yaml", ".cff"}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if (
            not path.is_file()
            or ".git" in path.parts
            or relative.parts[0] in {"data", "outputs"}
            or path.suffix not in extensions
        ):
            continue
        text = path.read_text()
        for value in forbidden:
            assert value not in text, f"{value!r} remains in {path}"


def test_publication_metadata() -> None:
    license_text = (ROOT / "LICENSE").read_text()
    citation_text = (ROOT / "CITATION.cff").read_text()
    readme_text = (ROOT / "README.md").read_text()

    assert license_text.startswith("MIT License")
    assert "license: MIT" in citation_text
    assert "https://doi.org/10.5281/zenodo.10554304" in readme_text
    assert "dataset-detail?ds=DTS23011" in readme_text
    assert "not expected to match" in readme_text


def test_portable_default_paths() -> None:
    env = os.environ.copy()
    for key in (
        "MUELLERPT_INPUT_ROOT",
        "MUELLERPT_OUTPUT_ROOT",
        "MUELLERPT_POLAMBRIMETRY_ROOT",
        "MUELLERPT_CHECKPOINT",
    ):
        env.pop(key, None)

    checks = (
        (
            ROOT / "pretraining",
            "import config_pretrain as c; print(c.OUTPUT_DIR)",
            ROOT / "outputs" / "results" / "pretraining",
        ),
        (
            ROOT / "experiments" / "colopola",
            "import config as c; print(c.TrainConfig().output_dir)",
            ROOT / "outputs" / "results" / "colopola",
        ),
        (
            ROOT / "experiments" / "polambrimetry",
            "import config as c; print(c.POLAMBRIMETRY_ROOT)",
            ROOT / "data" / "polambrimetry",
        ),
    )
    for cwd, code, expected in checks:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip().splitlines()[-1] == str(expected)


def test_paper_launchers() -> None:
    launchers = (
        ROOT / "scripts" / "reproduce_colopola_table.sh",
        ROOT / "scripts" / "reproduce_polambrimetry_table.sh",
    )
    for launcher in launchers:
        subprocess.run(["bash", "-n", str(launcher)], check=True)

    colopola = launchers[0].read_text()
    seed_block = re.search(
        r"--split-seeds \\\n(?P<body>.*?)  --few-shot-percentages",
        colopola,
        re.DOTALL,
    )
    assert seed_block is not None
    seeds = [int(value) for value in re.findall(r"\b\d+\b", seed_block.group("body"))]
    assert seeds == [
        42, 400, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
        10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008,
        10009, 10010, 20020, 30030, 40040, 50050, 60060, 70070,
        80080, 90090, 101010, 101011, 101012,
    ]

    polambrimetry = launchers[1].read_text()
    assert "--few-shot-percentages 1 5 25 50 100" in polambrimetry
    assert "--resume" in polambrimetry


def test_downstream_code_contains_only_paper_variants() -> None:
    polam_files = (
        ROOT / "experiments" / "polambrimetry" / "config.py",
        ROOT / "experiments" / "polambrimetry" / "model.py",
        ROOT / "experiments" / "polambrimetry" / "train_unet.py",
    )
    colopola_files = (
        ROOT / "experiments" / "colopola" / "config.py",
        ROOT / "experiments" / "colopola" / "dataset.py",
        ROOT / "experiments" / "colopola" / "train_unet.py",
    )

    polam_text = "\n".join(path.read_text() for path in polam_files)
    for removed in (
        "DecompPredictor",
        "HRNetDecompHead",
        "decomp_checkpoint",
        "_load_sweep_grid",
        "_grid_product",
    ):
        assert removed not in polam_text

    colopola_text = "\n".join(path.read_text() for path in colopola_files)
    for removed in (
        "input_channel_preset",
        "channel_keep_mask_from_preset",
        "reestimate_bn_stats",
        "freeze_encoder",
        "random_init_only",
        "_load_sweep_spec",
        "_expand_sweep_grid",
    ):
        assert removed not in colopola_text

    assert "matplotlib" not in polam_text
    assert "matplotlib" not in colopola_text


def test_pretraining_contains_only_published_protocol() -> None:
    pretraining = "\n".join(
        path.read_text()
        for path in (ROOT / "pretraining").rglob("*.py")
    )
    for removed in (
        "CONFIG_PRETRAIN_DIR",
        "PRETRAIN_VAL_FRACTION",
        "INPUT_NORM_MODE",
        "INPUT_PATCH_SIZE",
        "ROTATION_VARIANT",
        "CHANNEL_DROP_MODE",
        "make_patch_mask",
        "make_block_mask",
        "make_channel_mask_count",
        "build_decomp_targets",
        "matplotlib",
    ):
        assert removed not in pretraining

    config_path = ROOT / "pretraining" / "config_pretrain.py"
    namespace: dict[str, object] = {"__file__": str(config_path)}
    exec(compile(config_path.read_text(), str(config_path), "exec"), namespace)
    assert namespace["DATASET_NAME"] == "0902Measurement"
    assert namespace["CENTER_CROP_SIZE"] == (600, 600)
    assert namespace["EPOCHS"] == 150
    assert namespace["DECOMP_ORDER"] == (
        "retardance",
        "depolarization",
        "diattenuation",
    )
    assert namespace["CHANNEL_DROP_KEEP_FULL_PROB"] == 0.5
    assert namespace["M00_DROP_PROB"] == 0.3

    removed_files = (
        ROOT / "pretraining" / "pretrain_vis.py",
        ROOT / "pretraining" / "pretrain_model_utils.py",
        ROOT / "pretraining" / "utils" / "compute_norm_stats.py",
    )
    assert not any(path.exists() for path in removed_files)


def test_shared_pretraining_models_construct_downstream_models() -> None:
    checks = (
        (
            ROOT / "experiments" / "polambrimetry",
            "import torch; from model import HRNetSegmentation; "
            "m=HRNetSegmentation(use_m00=True); "
            "assert m.encoder.highres_channels == 18; "
            "x=torch.randn(1,16,64,64); "
            "y=m(x,torch.rand(1,1,64,64),torch.ones(1,dtype=torch.bool)); "
            "assert y.requires_grad; y.mean().backward(); "
            "assert any(p.grad is not None for p in m.parameters() if p.requires_grad)",
        ),
        (
            ROOT / "experiments" / "colopola",
            "from model import HRNetClassifier; "
            "m=HRNetClassifier(); "
            "assert m.encoder.highres_channels == 18",
        ),
    )
    for cwd, code in checks:
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )


def test_reference_results_are_aggregate_and_sanitized() -> None:
    reference_root = ROOT / "results" / "reference_tables"
    expected_rows = {
        "colopola_classification.csv": 5,
        "polambrimetry_overall_dice.csv": 5,
        "polambrimetry_gm_wm_dice.csv": 10,
    }
    forbidden_columns = {
        "run_dir", "seed_sweep_dir", "specimen_id", "test_specimen",
        "val_specimen", "path", "filename",
    }
    for filename, row_count in expected_rows.items():
        with (reference_root / filename).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == row_count
        assert forbidden_columns.isdisjoint(rows[0])
        assert not any("/home/" in value or "/mnt/" in value for row in rows for value in row.values())


def test_reference_tables_match_publication_scope() -> None:
    reference_root = ROOT / "results" / "reference_tables"
    with (reference_root / "colopola_classification.csv").open(newline="") as stream:
        colopola = list(csv.DictReader(stream))
    assert set(colopola[0]) == {
        "few_shot_pct",
        "n_seeds_pretrained",
        "n_seeds_random",
        "pretrained_test_acc",
        "random_test_acc",
        "pretrained_test_recall",
        "random_test_recall",
        "pretrained_test_specificity",
        "random_test_specificity",
    }
    assert all(row["n_seeds_pretrained"] == "30" for row in colopola)
    assert all(row["n_seeds_random"] == "30" for row in colopola)

    with (reference_root / "polambrimetry_overall_dice.csv").open(newline="") as stream:
        overall = list(csv.DictReader(stream))
    with (reference_root / "polambrimetry_gm_wm_dice.csv").open(newline="") as stream:
        per_class = list(csv.DictReader(stream))
    class_by_key = {
        (row["few_shot_pct"], row["class_name"]): row for row in per_class
    }
    for row in overall:
        pct = row["few_shot_pct"]
        for prefix in ("pretrained", "random"):
            gm = float(class_by_key[(pct, "gm")][f"{prefix}_mean_dice"])
            wm = float(class_by_key[(pct, "wm")][f"{prefix}_mean_dice"])
            reported = float(row[f"{prefix}_mean_dice"])
            assert abs(reported - (gm + wm) / 2.0) < 1e-6
            assert row[f"{prefix}_n"] == "30"


def test_publication_aggregators_have_no_stale_refactor_names() -> None:
    polam = (
        ROOT / "experiments" / "polambrimetry" / "summarize_metrics.py"
    ).read_text()
    colopola = (
        ROOT / "experiments" / "colopola" / "aggregate_seed_results.py"
    ).read_text()
    assert "_class_names" not in polam
    assert "delta_pretrained_minus_random" not in colopola


def test_polambrimetry_summarizer_runs(tmp_path: Path) -> None:
    input_csv = tmp_path / "nested_summary.csv"
    input_csv.write_text(
        "run,few_shot_pct,test_mean_dice,test_dice_gm,test_dice_wm,"
        "pair_index,outer_fold,test_specimen,val_specimen\n"
        "ssl_pretrained,1,0.60,0.80,0.40,0,0,s1,s2\n"
        "random,1,0.50,0.70,0.30,0,0,s1,s2\n"
    )
    output_dir = tmp_path / "summary"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "polambrimetry" / "summarize_metrics.py"),
            "--input-csv",
            str(input_csv),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with (output_dir / "few_shot_dice_summary_wide.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "few_shot_pct": "1",
            "pretrained_mean_dice": "0.600000",
            "pretrained_std_dice": "0.000000",
            "pretrained_n": "1",
            "random_mean_dice": "0.500000",
            "random_std_dice": "0.000000",
            "random_n": "1",
        }
    ]
