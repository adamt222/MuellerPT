from __future__ import annotations

from datetime import datetime
from pathlib import Path

from config_pretrain import OUTPUT_DIR, RUN_NAME


def init_run_dir() -> tuple[str, Path]:
    run_id = RUN_NAME or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def write_run_config(run_dir: Path) -> None:
    source = Path(__file__).resolve().parent / "config_pretrain.py"
    (run_dir / "config_pretrain.py").write_text(source.read_text())


def append_metrics(
    run_dir: Path,
    run_id: str,
    epoch: int,
    loss: float,
) -> None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text("epoch,loss,decomp\n")
    with metrics_path.open("a") as stream:
        stream.write(f"{epoch},{loss:.6f},{loss:.6f}\n")

    summary_path = OUTPUT_DIR / "summary.csv"
    if not summary_path.exists():
        summary_path.write_text("run_id,epoch,loss,decomp\n")
    with summary_path.open("a") as stream:
        stream.write(f"{run_id},{epoch},{loss:.6f},{loss:.6f}\n")
