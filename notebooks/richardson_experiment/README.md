# Richardson experiment (v1 → v5)

Richardson's iterative ablation notebooks — the full model-development story from scratch dual-encoder through caption augmentation, rotation, and pretrained text encoders.

These notebooks are **integrated with the main repo**: they read from `data/processed/`, write checkpoints to `models/experiments/`, and share the same Kaggle dataset layout as `src/prepare_data.py`.

## How this fits the project narrative

| Step | Notebook | What changed | Version |
|---|---|---|---|
| Explore + preprocess | `01_explore_and_preprocess.ipynb` | Dataset EDA, caption templates, train/val/test splits | — |
| Baseline dual-encoder | `02_train_dual_encoder.ipynb` | Scratch text encoder, templated captions | **v1** |
| Fine-tune + compare | `03_finetune_and_compare.ipynb` | Fine-tune image encoder vs baseline | v1 ft |
| Full dataset | `04_final_full_dataset.ipynb` | Scale to full ~44k products | v1 full |
| Caption augmentation | `05_caption_augmentation.ipynb` | 4 query styles (templated, short, shopper, brand) | **v2** |
| Caption rotation | `06_caption_rotation.ipynb` | Random query style per epoch during training | **v3** |
| Pretrained text | `07_pretrained_text_encoder.ipynb` | Frozen MiniLM text encoder — **selected final model** | **v4** |
| Pretrained + rotation | `08_pretrained_with_rotation.ipynb` | v4 + caption rotation (did not beat v4) | **v5** |

After v4 was locked, Samii refactored training/evaluation into `src/` and reproduced results in `samii_experiment/`:

- `notebooks/samii_experiment/03_train_dual_encoder.ipynb` → `python src/train_dual_encoder.py`
- `notebooks/samii_experiment/04_final_results.ipynb` → `python src/evaluate_all.py`

## Artifacts

| Location | Contents |
|---|---|
| `models/experiments/` | Per-version weights, loss curves, comparison plots |
| `docs/reports/ablations/` | Recall@K comparison CSVs (v1–v5) |
| `data/processed/` | Splits + augmented long-form CSVs (`train_aug.csv`, etc.) |

Key comparison table: `docs/reports/ablations/v1_v2_v3_v4_comparison.csv`

Final unified test-set evaluation (TF-IDF vs dual-encoder): `docs/reports/evaluation_results.csv`

## Run order

1. Download data: `python src/download_kaggle_data.py`
2. Preprocess: `python src/prepare_data.py` (or walk through `01_explore_and_preprocess.ipynb`)
3. Run experiment notebooks **in order** (02 → 08) to reproduce the ablation path
4. Train final v4 via CLI: `python src/train_dual_encoder.py`
5. Evaluate: `python src/evaluate_all.py`
6. View combined story: `notebooks/samii_experiment/04_final_results.ipynb`
