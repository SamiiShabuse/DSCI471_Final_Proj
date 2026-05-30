# Reproducible Artifacts

All paths are defined in `src/paths.py`.

## Data (`data/processed/`)

| File | Description |
|---|---|
| `pairs.csv` | All 44,265 products |
| `train.csv` | Training split (35,412) |
| `val.csv` | Validation split (4,426) |
| `test.csv` | Test split (4,427) — evaluation gallery |
| `products.csv` | TF-IDF index (`product_text` with JSON descriptions) |
| `train_aug.csv`, `val_aug.csv`, `test_aug.csv` | Long-form augmented splits (4 query styles per product) |

Regenerate: `python src/prepare_data.py` (base splits); augmented CSVs from `notebooks/richardson_experiment/05_caption_augmentation.ipynb`

## Final model (`models/`)

**Not in git** — regenerate locally (see [`models/README.md`](../models/README.md)).

| File | Description |
|---|---|
| `v4_image_encoder.weights.h5` | **Canonical** final EfficientNetB0 image tower |
| `embeddings/train_text.npy` | Cached MiniLM train caption embeddings |
| `embeddings/val_text.npy` | Cached MiniLM val caption embeddings |
| `test_image_embeddings.npy` | Cached test gallery image embeddings |

```powershell
python src/train.py      # creates weights + train/val text embeddings
python src/evaluate.py   # creates test image embeddings + evaluation_results.csv
```

## Ablation experiments (`models/experiments/`)

**Not in git** — optional; reproduce via Richardson notebooks. Metrics CSVs **are** in git under `docs/reports/ablations/`.

See `notebooks/richardson_experiment/README.md` for the notebook mapping.

## Results (`docs/reports/`)

| File | Description |
|---|---|
| `evaluation_results.csv` | Final TF-IDF vs dual-encoder comparison (test set) |
| `ablations/*.csv` | Ablation Recall@K tables (single source of truth) |

Regenerate final results: `python src/evaluate.py` (full test set)

Smoke test: `python src/evaluate.py --sample 500` → `evaluation_results_sample.csv`

Grading workflow: `docs/GRADING.md`
