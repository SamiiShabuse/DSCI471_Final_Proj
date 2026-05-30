# Reports and Results

| File | Description |
|---|---|
| `evaluation_results.csv` | Final unified comparison: TF-IDF vs dual-encoder on test set |
| `ablations/` | Ablation Recall@K metrics (CSVs only — see `ablations/README.md`) |

Notebooks that produced these results: `notebooks/richardson_experiment/` (see `notebooks/richardson_experiment/README.md`).

## Regenerate final results

```powershell
python src/evaluate.py
```

## Ablation files

These CSVs document the iterative model development that led to choosing v4 as the final architecture. They use Recall@K on the validation set from earlier experiments and are referenced in the final report.
