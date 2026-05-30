"""Compare two evaluation_results CSV files."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compare(official_path: Path, rerun_path: Path) -> None:
    official = pd.read_csv(official_path)
    rerun = pd.read_csv(rerun_path)
    merged = official.merge(rerun, on=["Model", "Query type"], suffixes=("_official", "_rerun"))
    metrics = ["Top-1", "Top-5", "MRR", "Precision@5"]

    print("=== Comparison: official vs fresh rerun ===\n")
    max_abs_delta = 0.0
    for _, row in merged.iterrows():
        print(f"{row['Model']} | {row['Query type']}")
        for m in metrics:
            o = row[f"{m}_official"]
            r = row[f"{m}_rerun"]
            diff = r - o
            max_abs_delta = max(max_abs_delta, abs(diff))
            if abs(diff) < 1e-9:
                note = "EXACT"
            else:
                note = f"delta={diff:+.4f}"
            print(f"  {m}: official={o:.4f}  rerun={r:.4f}  ({note})")
        print()

    print(f"Max absolute delta across all metrics: {max_abs_delta:.6f}")


if __name__ == "__main__":
    a = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "docs/reports/evaluation_results_official_backup.csv"
    b = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "docs/reports/evaluation_results_rerun.csv"
    compare(a, b)
