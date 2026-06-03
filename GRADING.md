# Grading and Reproduction Guide

For instructors and anyone re-running this project.

## What Is in Git vs Generated Locally

| In git | Generated locally |
|---|---|
| Source code in `src/` | `data/raw/` and `data/processed/` |
| Notebooks | `models/` weights and embeddings |
| Final result CSVs in `docs/reports/` | Optional smoke-test CSVs |
| Report figures in `docs/reports/figures/` | Local model caches |

Precomputed final results are committed at `docs/reports/evaluation_results.csv`
for the full 4,427-product test set.

## One-Time Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/download_kaggle_data.py
python src/prepare_data.py
python src/prepare_data.py --check
```

Kaggle download requires Kaggle API credentials. The first dual-encoder training
or inference run also requires internet/Wi-Fi so `sentence-transformers` can
load `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face. If MiniLM is
already cached locally, the dual-encoder path can run offline.

If graders need dual-encoder inference without network access or retraining,
submit these artifacts separately:

- `models/v4_image_encoder.weights.h5`
- `models/test_image_embeddings.npy`
- cached MiniLM / Hugging Face model artifacts, if the environment is offline

## Recommended Test Sequence

Use sample/smoke commands for quick verification and full commands for final
numbers.

### 1. Quick Pipeline Check

```powershell
python src/evaluate.py --baseline-only
```

This requires only `data/processed/` and confirms the TF-IDF path and data schema.

For a faster smoke check that does not touch the official results file:

```powershell
python src/prepare_data.py --check
python src/evaluate.py --baseline-only --sample 100 --output docs/reports/evaluation_results_smoke_check.csv
```

This validates the processed split schema and writes a small local smoke-test CSV.
The official full-test metrics remain in `docs/reports/evaluation_results.csv`.

### 2. Smoke Test With Subsample

```powershell
python src/evaluate.py --sample 500
```

This writes `docs/reports/evaluation_results_sample.csv` and does not overwrite
the official `evaluation_results.csv`.

```powershell
python src/train.py --sample 800 --baseline-epochs 1 --finetune-epochs 1
python src/evaluate.py --sample 500 --dual-only
```

Sample training creates real but small weights at
`models/v4_image_encoder.weights.h5`. Re-run full `python src/train.py` for
submission-quality model weights.

### 3. Full Reproduction

```powershell
python src/train.py
python src/evaluate.py
```

This regenerates model weights, embeddings, and
`docs/reports/evaluation_results.csv`. CPU training is expected to take roughly
40 minutes or more depending on hardware.

## Official vs Sample Outputs

| Command | Output file | Purpose |
|---|---|---|
| `python src/evaluate.py` | `evaluation_results.csv` | Official full test-set metrics |
| `python src/evaluate.py --sample N` | `evaluation_results_sample.csv` | Quick smoke test |
| `python src/evaluate.py --output path.csv` | Custom path | Ad-hoc experiments |

## Notebooks

```powershell
jupyter lab notebooks/
```

- `samii_experiment/`: final pipeline demos using `src/`
- `richardson_experiment/`: v1-to-v5 development history

## Troubleshooting

| Error | Fix |
|---|---|
| Missing processed files or missing `product_text` | Run `python src/prepare_data.py` |
| Dual-encoder weights not found | Run `python src/train.py`, or use `--baseline-only` |
| MiniLM / Hugging Face download fails | Connect to internet/Wi-Fi once, then rerun; or pre-cache the Hugging Face MiniLM artifacts |
| Slow training on Windows | Expected on CPU; WSL2/Colab GPU is optional |

## Submission Artifacts

For grading without retraining, the committed files in `docs/reports/` are the
main evidence:

- `evaluation_results.csv`: final TF-IDF vs dual-encoder comparison
- `ablations/*.csv`: validation ablation progression
- `figures/*.png`: EDA, metric, ablation, and qualitative retrieval figures
- `final_report.pdf`: print-ready final report
- `final_report.md`: source report

Optional but recommended if graders should run dual-encoder demos immediately:

- `models/v4_image_encoder.weights.h5`
- `models/test_image_embeddings.npy`
