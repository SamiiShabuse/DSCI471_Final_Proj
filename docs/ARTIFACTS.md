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

## Experiment artifacts (`models/experiments/`)

| File | Description |
|---|---|
| `v*_image_encoder.weights.h5` | Per-version image encoder weights from ablation runs |
| `v1_v2_v3_v4_comparison.csv` | Recall@K across v1–v4 (also in `docs/reports/ablations/`) |
| `final_comparison.csv` | v4 vs v5 per query style |
| `*_loss_curve.png`, `*_recall*.png` | Training and comparison plots |

See `notebooks/richardson_experiment/README.md` for the full v1→v5 notebook mapping.

## Model (`models/`)

| File | Description |
|---|---|
| `v4_image_encoder.weights.h5` | Final EfficientNetB0 image tower |
| `embeddings/train_text.npy` | Cached MiniLM train caption embeddings |
| `embeddings/val_text.npy` | Cached MiniLM val caption embeddings |
| `test_image_embeddings.npy` | Cached test gallery image embeddings |

Regenerate weights: `python src/train_dual_encoder.py`

## Results (`docs/reports/`)

| File | Description |
|---|---|
| `evaluation_results.csv` | Final TF-IDF vs dual-encoder comparison |
| `ablations/*.csv` | v1–v4 iteration metrics for the report |

Regenerate: `python src/evaluate_all.py`
