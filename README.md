# Multimodal Deep Learning Search Engine for E-commerce Fashion

DSCI471 final project (Richardson Chhin, Samii Shabuse): a multimodal dual-encoder retrieval system for fashion product search. Natural-language queries retrieve product images via a shared embedding space, evaluated against a TF-IDF keyword baseline.

## Research question

Can a dual-encoder deep learning model, trained on paired fashion images and text, outperform traditional keyword-based e-commerce search?

## Final model (v4)

- **Image encoder:** EfficientNetB0 (ImageNet pretrained, fine-tuned) → 384-d embedding
- **Text encoder:** Frozen `sentence-transformers/all-MiniLM-L6-v2`
- **Training:** Contrastive loss
- **Retrieval:** Cosine similarity (text query → image gallery)

## Project structure

```
DSCI471_Final_Proj/
├── data/
│   ├── raw/fashion-dataset/       # Kaggle dataset (not in git)
│   └── processed/                 # Generated splits
├── docs/
│   ├── DSCI471 Project Proposal.md
│   ├── Project_Finishing_Plan.md
│   ├── ARTIFACTS.md
│   └── reports/
│       ├── evaluation_results.csv
│       └── ablations/             # Ablation metrics CSVs
├── models/
│   ├── v4_image_encoder.weights.h5   # Final model (canonical)
│   ├── experiments/                  # Ablation weights & plots only
│   └── embeddings/
├── notebooks/
│   ├── richardson_experiment/     # v1→v5 ablation notebooks
│   └── samii_experiment/          # Final pipeline notebooks
└── src/
    ├── paths.py                   # Directories and artifact paths
    ├── config.py                  # Hyperparameters and constants
    ├── captions.py                # Caption + query generators
    ├── prepare_data.py            # Data preprocessing CLI
    ├── download_kaggle_data.py    # Dataset download CLI
    ├── model.py                   # Dual-encoder architecture
    ├── baseline_keyword.py        # TF-IDF baseline search
    ├── search.py                  # Load models and run retrieval
    ├── metrics.py                 # Top-K, MRR, Precision@K
    ├── train.py                   # Training CLI
    └── evaluate.py                # Unified evaluation CLI
```

## Quickstart

### 1. Environment

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download data

Place the [Fashion Product Images dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) under `data/raw/fashion-dataset/`, or:

```powershell
python src/download_kaggle_data.py
```

### 3. Preprocess

```powershell
python src/prepare_data.py
```

### 4. Train

```powershell
python src/train.py
```

Smoke test: `python src/train.py --sample 800 --baseline-epochs 1 --finetune-epochs 1`

### 5. Evaluate

```powershell
python src/evaluate.py
```

### 6. Notebooks

```powershell
jupyter lab notebooks/
```

## Results (test set, 4,427 products)

| Query type | TF-IDF Top-1 | Dual-encoder Top-1 |
|---|---|---|
| templated | 0.52 | 0.17 |
| shopper | 0.08 | 0.07 |
| brand | 0.65 | 0.14 |
| short | 0.07 | 0.05 |

Full metrics: `docs/reports/evaluation_results.csv`

## Troubleshooting

- **Missing data:** Run `prepare_data.py` after downloading Kaggle files.
- **Missing weights:** Run `train.py` before full evaluation.
- **Slow on Windows:** TensorFlow uses CPU; use Colab/WSL2 with GPU for training.

## Credits

DSCI471 course project. Dataset: [Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) (Aggarwal, Kaggle).
