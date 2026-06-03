"""Run a quick sample training + evaluation pipeline (smoke test).

This script runs `src/train.py` and `src/evaluate.py` in sample mode
so graders can reproduce a small, fast run without full training.
"""
import subprocess
import sys
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ARTIFACTS_TO_PROTECT = [
    ROOT / "models" / "v4_image_encoder.weights.h5",
    ROOT / "models" / "test_image_embeddings.npy",
    ROOT / "models" / "embeddings" / "train_text.npy",
    ROOT / "models" / "embeddings" / "val_text.npy",
]

def run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def backup_artifacts() -> tuple[Path, set[Path]]:
    backup_dir = ROOT / "models" / ".sample_pipeline_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    existed = set()
    for artifact in ARTIFACTS_TO_PROTECT:
        if artifact.exists():
            existed.add(artifact)
            dest = backup_dir / artifact.relative_to(ROOT / "models")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact, dest)
    return backup_dir, existed


def restore_artifacts(backup_dir: Path, existed: set[Path]) -> None:
    for artifact in ARTIFACTS_TO_PROTECT:
        backup = backup_dir / artifact.relative_to(ROOT / "models")
        if artifact in existed and backup.exists():
            artifact.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, artifact)
        elif artifact.exists():
            artifact.unlink()
    shutil.rmtree(backup_dir, ignore_errors=True)

def main():
    backup_dir, existed = backup_artifacts()
    try:
        # 1) Sample train (small, fast). This writes canonical artifact paths,
        # so restore_artifacts puts any full-run files back afterward.
        run([PY, "src/train.py", "--sample", "800", "--baseline-epochs", "1", "--finetune-epochs", "1"])

        # 2) Sample evaluation (does not overwrite official results)
        run([PY, "src/evaluate.py", "--sample", "500", "--output", "docs/reports/evaluation_results_sample.csv"])
    finally:
        restore_artifacts(backup_dir, existed)

    print("Sample pipeline finished. See docs/reports/evaluation_results_sample.csv")

if __name__ == "__main__":
    main()
