# Grading & reproduction guide

For instructors and anyone re-running this project multiple times.

## What is in git vs generated locally

| In git | Generated locally (see below) |
|---|---|
| Source code (`src/`) | `data/raw/` and `data/processed/` |
| Notebooks | `models/` weights and embeddings |
| Result CSVs (`docs/reports/`) | Optional smoke-test CSV |

Pre-computed **final results** are committed at `docs/reports/evaluation_results.csv` (full test set, 4,427 products). You can reproduce them with the steps below.

## One-time setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/download_kaggle_data.py
python src/prepare_data.py
python src/prepare_data.py --check
```

Kaggle download requires [Kaggle API credentials](https://github.com/Kaggle/kagglehub).

## Recommended test sequence

Run in order. Use **sample/smoke** commands for quick verification; use **full** commands for final numbers.

### 1. Quick pipeline check (no training, ~1 min)

```powershell
python src/evaluate.py --baseline-only
```

Requires only `data/processed/` — confirms TF-IDF path and data schema.

### 2. Smoke test with subsample (does not overwrite official results)

```powershell
python src/evaluate.py --sample 500
```

Writes to `docs/reports/evaluation_results_sample.csv` only. **`evaluation_results.csv` is not modified.**

```powershell
python src/train.py --sample 800 --baseline-epochs 1 --finetune-epochs 1
python src/evaluate.py --sample 500 --dual-only
```

Sample training creates real but small weights at `models/v4_image_encoder.weights.h5` (overwrites local full weights if present — re-run full `train.py` for submission-quality model).

### 3. Full reproduction (~40+ min CPU for training)

```powershell
python src/train.py
python src/evaluate.py
```

Regenerates `models/v4_image_encoder.weights.h5`, embeddings, and `docs/reports/evaluation_results.csv`.

## Official vs sample outputs

| Command | Output file | Purpose |
|---|---|---|
| `python src/evaluate.py` | `evaluation_results.csv` | **Official** full test-set metrics |
| `python src/evaluate.py --sample N` | `evaluation_results_sample.csv` | Quick smoke test only |
| `python src/evaluate.py --output path.csv` | Custom path | Ad-hoc experiments |

## Notebooks

```powershell
jupyter lab notebooks/
```

- `samii_experiment/` — final pipeline demos (uses `src/`)
- `richardson_experiment/` — v1→v5 development history

## Troubleshooting

| Error | Fix |
|---|---|
| `Missing processed files` / `missing columns ['product_text']` | `python src/prepare_data.py` |
| `Dual-encoder weights not found` | `python src/train.py` (or use `--baseline-only`) |
| Slow training on Windows | Expected on CPU; use Colab/WSL2 with GPU optional |

## Submission artifacts

For grading without retraining, the committed files in `docs/reports/` are sufficient:

- `evaluation_results.csv` — final TF-IDF vs dual-encoder comparison
- `ablations/*.csv` — Richardson experiment progression (validation Recall@K)

Optional: attach a zip of `models/v4_image_encoder.weights.h5` if dual-encoder demo is required without retraining.
