# Notebooks

Two teammate folders, one shared pipeline via `src/`:

```powershell
jupyter lab notebooks/
```

## Samii — unified pipeline (`samii_experiment/`)

Submission-ready reproduction of the final v4 model and evaluation.

| # | Notebook | Purpose |
|---|---|---|
| 01 | `samii_experiment/01_explore_data.ipynb` | Dataset overview and sample images |
| 02 | `samii_experiment/02_keyword_baseline_demo.ipynb` | TF-IDF search demo |
| 03 | `samii_experiment/03_train.ipynb` | Final v4 training walkthrough |
| 04 | `samii_experiment/04_final_results.ipynb` | Phase 4 results, demos, failure analysis |

## Richardson — model development (`richardson_experiment/`)

Iterative ablation notebooks (v1 → v5). See [`richardson_experiment/README.md`](richardson_experiment/README.md).

| # | Notebook | Version |
|---|---|---|
| 01 | `richardson_experiment/01_explore_and_preprocess.ipynb` | Data EDA + splits |
| 02 | `richardson_experiment/02_train_dual_encoder.ipynb` | v1 — scratch text encoder |
| 03 | `richardson_experiment/03_finetune_and_compare.ipynb` | v1 fine-tuning |
| 04 | `richardson_experiment/04_final_full_dataset.ipynb` | v1 at full scale |
| 05 | `richardson_experiment/05_caption_augmentation.ipynb` | v2 — 4 query styles |
| 06 | `richardson_experiment/06_caption_rotation.ipynb` | v3 — caption rotation |
| 07 | `richardson_experiment/07_pretrained_text_encoder.ipynb` | **v4 — final model** |
| 08 | `richardson_experiment/08_pretrained_with_rotation.ipynb` | v5 — v4 + rotation |

## CLI equivalents

| Task | Command |
|---|---|
| Preprocess data | `python src/prepare_data.py` |
| Train model | `python src/train.py` |
| Evaluate | `python src/evaluate.py` |
| Download data | `python src/download_kaggle_data.py` |
