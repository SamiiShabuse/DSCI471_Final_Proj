# Reports and Results

| File | Description |
|---|---|
| `evaluation_results.csv` | **Official** full test-set comparison (TF-IDF vs dual-encoder) |
| `evaluation_results_sample.csv` | Smoke-test output from `evaluate.py --sample` (gitignored, local only) |
| `final_report.md` | **Final written report** |
| `ablations/` | Ablation Recall@K metrics (CSVs only — see `ablations/README.md`) |

Notebooks that produced ablation results: `notebooks/richardson_experiment/`

## Regenerate official results

Full test gallery (4,427 products):

```powershell
python src/prepare_data.py --check
python src/evaluate.py
```

Quick smoke test (does not overwrite `evaluation_results.csv`):

```powershell
python src/evaluate.py --sample 500
```

See [`docs/GRADING.md`](../GRADING.md) for the full grading workflow.

## Ablation files

These CSVs document the iterative model development that led to choosing v4 as the final architecture. They use Recall@K on the validation set from Richardson's experiment notebooks.
