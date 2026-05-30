# Models directory

Generated artifacts live here and are **not stored in git** (too large). Small result tables are in `docs/reports/`.

## Layout

```
models/
├── v4_image_encoder.weights.h5      # Final model (required for evaluate.py)
├── test_image_embeddings.npy        # Cached test gallery embeddings
├── embeddings/
│   ├── train_text.npy               # Cached MiniLM train captions
│   └── val_text.npy                 # Cached MiniLM val captions
└── experiments/                     # Richardson ablation weights & plots (optional)
```

## Regenerate (required for full pipeline)

From the project root with `venv` activated and `data/processed/` present:

```powershell
# 1. Train final v4 image encoder (~40 min on CPU; faster with GPU)
python src/train.py

# 2. Run evaluation (builds test_image_embeddings.npy if missing)
python src/evaluate.py
```

Smoke test (small subset):

```powershell
python src/train.py --sample 800 --baseline-epochs 1 --finetune-epochs 1
python src/evaluate.py --sample 500
```

## Ablation artifacts (`experiments/`)

Only needed to reproduce Richardson's v1→v5 notebook story. Re-run notebooks in `notebooks/richardson_experiment/` (02→08) or skip if you only need the final model.

Metrics from those runs are already saved as CSVs in `docs/reports/ablations/` (those **are** in git).

## Course submission

If the grader cannot retrain:

1. **Preferred:** Submit this repo plus a separate zip of `models/v4_image_encoder.weights.h5` and `models/test_image_embeddings.npy` (or full `models/`), linked in your report README.
2. **Alternative:** Note in the report that eval requires running `train.py` once; include `evaluation_results.csv` as the graded output.
