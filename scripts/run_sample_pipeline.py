"""Run a quick sample training + evaluation pipeline (smoke test).

This script runs `src/train.py` and `src/evaluate.py` in sample mode
so graders can reproduce a small, fast run without full training.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

def run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

def main():
    # 1) Sample train (small, fast)
    run([PY, "-m", "src.train", "--sample", "800", "--baseline-epochs", "1", "--finetune-epochs", "1"])

    # 2) Sample evaluation (does not overwrite official results)
    run([PY, "-m", "src.evaluate", "--sample", "500", "--output", "docs/reports/evaluation_results_sample.csv"])

    print("Sample pipeline finished. See docs/reports/evaluation_results_sample.csv")

if __name__ == "__main__":
    main()
