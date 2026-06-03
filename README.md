# Multimodal Deep Learning Search Engine for E-commerce Fashion

**DSCI471 Final Project**  
**Team:** Richardson Chhin and Samii Shabuse  
**Project type:** Multimodal deep learning retrieval system  
**Dataset:** Kaggle Fashion Product Images dataset  
**Final model:** EfficientNetB0 image encoder + frozen MiniLM text encoder  

---

## Executive Summary

This project builds a fashion product search engine that retrieves product images from natural-language text queries. The main research question is:

> **Can a dual-encoder deep learning model, trained on paired fashion images and text descriptions, outperform traditional keyword-based e-commerce search?**

Our answer is nuanced. On exact Top-1 retrieval, the TF-IDF keyword baseline wins because it searches rich product text, including product titles and JSON descriptions. The dual-encoder solves a harder task: matching text queries directly to image embeddings. Even though it loses Top-1 overall, it is competitive on shopper-style queries at Top-5 and MRR, which matters for real search interfaces that show several results at once.

The project includes the full pipeline:

1. Download and prepare the Kaggle fashion dataset.
2. Generate captions and query styles from product metadata.
3. Explore dataset scale, class imbalance, colors, image samples, and text lengths.
4. Train a multimodal dual-encoder with contrastive learning.
5. Compare against a strong TF-IDF baseline.
6. Evaluate with Top-1, Top-5, MRR, and Precision@5.
7. Analyze why each method succeeds or fails.
8. Provide reproducible scripts, notebooks, reports, figures, and presentation artifacts.

---

## Final Result in One Table

Final evaluation uses a held-out test gallery of **4,427 products**.

| Model | Query type | Top-1 | Top-5 | MRR | Precision@5 |
|---|---|---:|---:|---:|---:|
| TF-IDF | templated | **0.523** | **0.805** | **0.649** | **0.161** |
| TF-IDF | shopper | **0.084** | 0.216 | 0.161 | 0.043 |
| TF-IDF | brand | **0.651** | **0.891** | **0.756** | **0.178** |
| TF-IDF | short | **0.073** | **0.201** | **0.146** | **0.040** |
| Dual-encoder (v4) | templated | 0.174 | 0.462 | 0.314 | 0.092 |
| Dual-encoder (v4) | shopper | 0.074 | **0.241** | **0.167** | **0.048** |
| Dual-encoder (v4) | brand | 0.140 | 0.402 | 0.270 | 0.080 |
| Dual-encoder (v4) | short | 0.052 | 0.176 | 0.126 | 0.035 |

**Interpretation:** TF-IDF is best when product text is available. The dual-encoder is still useful when retrieval must happen against images, or when a hybrid system needs visual reranking.

![Retrieval metrics comparison](docs/reports/figures/test_metrics_comparison.png)

---

## Problem Motivation

E-commerce shoppers rarely search in perfect catalog language. They might type "men's navy shirt" rather than the exact product title. Traditional search systems often depend on keyword overlap, so they can fail when shoppers use natural phrasing or when catalog descriptions are incomplete.

Multimodal retrieval offers another path. A dual-encoder model learns two embedding functions:

- Text query -> text embedding
- Product image -> image embedding

If the embeddings are aligned, a text query can retrieve visually matching product images by cosine similarity. This is the same broad idea used in CLIP-style retrieval systems.

---

## Dataset

We use the [Fashion Product Images dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset). Each product can include:

- Product image: `images/{id}.jpg`
- Product metadata: `styles.csv`
- Optional JSON description: `styles/{id}.json`
- Fields such as gender, master category, subcategory, article type, base color, season, usage, and display name

The preprocessing pipeline is implemented in [`src/prepare_data.py`](src/prepare_data.py).

### Processed Dataset Statistics

| Statistic | Value |
|---|---:|
| Products after filtering | 44,265 |
| Train products | 35,412 |
| Validation products | 4,426 |
| Test products | 4,427 |
| Master categories | 6 |
| Article types | 107 |
| Base colors | 46 |
| Products with loaded JSON descriptions | 44,136 |
| Median `product_text` length | 674 characters |
| Mean `product_text` length | 638 characters |

The largest article types are T-shirts, Shirts, Casual Shoes, Watches, Sports Shoes, Kurtas, Tops, Handbags, Heels, and Sunglasses. The largest master categories are Apparel, Accessories, and Footwear.

![Article distribution](docs/reports/figures/eda_article_distribution.png)

![Master category distribution](docs/reports/figures/eda_master_category_distribution.png)

![Color distribution](docs/reports/figures/eda_color_distribution.png)

![Representative catalog images](docs/reports/figures/eda_sample_images_grid.png)

![Product text length distribution](docs/reports/figures/eda_text_length_distribution.png)

---

## Preprocessing

The preprocessing script performs the following steps:

1. Loads product rows from `styles.csv`.
2. Keeps rows with matching image files.
3. Loads optional JSON descriptions.
4. Creates generated captions from structured attributes.
5. Builds `product_text` for TF-IDF by combining captions and descriptions.
6. Drops rare `articleType` classes with fewer than 10 examples.
7. Creates stratified 80/10/10 train/validation/test splits using random seed 42.

Generated files:

| File | Purpose |
|---|---|
| `data/processed/products.csv` | Full processed catalog |
| `data/processed/pairs.csv` | Image-text pairs |
| `data/processed/train.csv` | Training split |
| `data/processed/val.csv` | Validation split |
| `data/processed/test.csv` | Test split and final retrieval gallery |
| `data/processed/*_aug.csv` | Augmented query-style splits |

---

## Query Styles

Each test product generates four query styles. These make evaluation more realistic than using only one caption format.

| Query style | Meaning | Example |
|---|---|---|
| `templated` | Full attribute caption | `A men navy blue shirts, for casual wear in fall. Turtle Check Men Navy Blue Shirt.` |
| `shopper` | Natural short phrase | `men's navy blue shirt` |
| `brand` | Product display name | `Turtle Check Men Navy Blue Shirt` |
| `short` | Color + article type | `navy shirt` |

The same generated queries are used for TF-IDF and the dual-encoder.

---

## Model Architecture

The final model is **v4**, selected from the v1-to-v5 ablation path.

| Component | Final choice |
|---|---|
| Image encoder | EfficientNetB0, ImageNet pretrained |
| Image input | 224 x 224 x 3 |
| Projection head | Dense(512, GELU) -> Dropout(0.1) -> Dense(384) |
| Text encoder | Frozen `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Similarity | Cosine similarity / normalized dot product |
| Loss | Symmetric contrastive loss |
| Temperature | 0.07 |
| Batch size | 64 |
| Training epochs | 4 frozen-backbone epochs + 3 fine-tuning epochs |

### Why These Choices?

**EfficientNetB0** gives a good accuracy-to-compute tradeoff. It is small enough to train on course hardware but strong enough to extract useful visual features from product images.

**MiniLM** gives compact sentence embeddings and dramatically improved validation retrieval compared with a scratch text encoder. It is frozen to reduce training cost and overfitting risk.

**The projection head** adapts ImageNet visual features into the same 384-dimensional space as MiniLM. GELU provides a smooth nonlinearity, Dropout(0.1) adds light regularization, and L2 normalization makes dot product retrieval equivalent to cosine similarity.

**Contrastive learning** trains each matching image-text pair to be closer than mismatched pairs inside the batch. Temperature 0.07 sharpens the ranking signal and follows the CLIP-style retrieval setup.

Implementation:

- [`src/model.py`](src/model.py)
- [`src/train.py`](src/train.py)
- [`src/config.py`](src/config.py)

---

## Baseline

The baseline is a strong TF-IDF search system:

- Vectorizer: `TfidfVectorizer`
- Stop words: English
- Max features: 50,000
- Indexed text: `product_text`
- Ranking: cosine similarity

This baseline is intentionally strong because it searches product text that includes generated captions and JSON descriptions. It represents a realistic catalog search system.

Implementation:

- [`src/baseline_keyword.py`](src/baseline_keyword.py)
- [`src/evaluate.py`](src/evaluate.py)

---

## Evaluation Metrics

The evaluation uses standard retrieval metrics:

| Metric | Meaning |
|---|---|
| Top-1 | Whether the exact product is ranked first |
| Top-5 | Whether the exact product appears in the first five results |
| MRR | Mean reciprocal rank of the exact product |
| Precision@5 | Fraction of the first five slots occupied by the exact target; with one target, this is 0.2 when found in top five |

Final evaluation uses the held-out test split only. Development ablations use the validation split.

---

## Ablation Study

The model was not chosen blindly. Richardson's experiment notebooks track a v1-to-v5 development path.

| Version | Main change | Outcome |
|---|---|---|
| v1 | Scratch text encoder, templated captions | First full dual-encoder baseline |
| v2 | Four caption/query styles | More realistic query diversity |
| v3 | Random query-style rotation | Did not clearly improve over v2 |
| v4 | Frozen MiniLM text encoder | Best validation Recall@K |
| v5 | v4 plus caption rotation | Did not beat v4 |

Validation results for templated queries:

| Model | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| v1 | 0.106 | 0.332 | 0.488 |
| v2 | 0.080 | 0.271 | 0.425 |
| v3 | 0.068 | 0.247 | 0.387 |
| **v4** | **0.208** | **0.523** | **0.668** |

![Ablation progression](docs/reports/figures/ablation_progression.png)

![v4 training loss](docs/reports/figures/v4_training_loss.png)

![v4 vs v5 by query style](docs/reports/figures/v4_vs_v5_per_style.png)

---

## Interpretation

TF-IDF wins Top-1 because it has direct access to rich product text. Brand queries are exact product names, and templated queries closely resemble the indexed captions. That gives the keyword baseline a major advantage.

The dual-encoder has to infer the match from pixels. Images usually show product type, shape, and color, but they often do not show brand name, exact title, season, usage label, or JSON description. That is why the dual-encoder loses heavily on brand queries.

This is not just overfitting. The v4 ablation improvement shows the model learned useful visual-text alignment. The ceiling comes from the data and evaluation setup:

- Many products share the same color and article type.
- The evaluation has only one exact positive product.
- Visually similar products are counted as wrong.
- Metadata that helps TF-IDF is not always visible in images.

The dual-encoder is still meaningful for image-forward retrieval, visual search, and hybrid reranking.

---

## Qualitative Examples

Successful dual-encoder retrieval:

![Successful templated query retrieval](docs/reports/figures/demo_success_templated.png)

Head-to-head brand query:

![Head-to-head brand query](docs/reports/figures/demo_head_to_head_brand.png)

Shopper-query failure:

![Shopper failure case](docs/reports/figures/demo_failure_shopper.png)

---

## Limitations

1. Queries are generated from metadata, not real user logs.
2. Only the exact product ID is treated as correct.
3. Some short queries are naturally ambiguous.
4. MiniLM is frozen rather than fashion-domain fine-tuned.
5. Training uses course-scale compute rather than large-batch pretraining.
6. Brand names and descriptions are not always visible in images.
7. Full reproduction requires Kaggle data, TensorFlow, sentence-transformers, and model weights.

---

## Future Work

- Fine-tune the text encoder on fashion captions and product queries.
- Add hard negative mining for visually similar products.
- Train with larger effective batch sizes.
- Use CLIP-style pretraining on fashion image-text pairs.
- Evaluate on real search logs or click data.
- Build a hybrid search system: TF-IDF shortlist -> dual-encoder reranking.

---

## Project Structure

```text
DSCI471_Final_Proj/
|-- data/
|   |-- raw/fashion-dataset/       # Kaggle dataset, not stored in git
|   |-- processed/                 # Generated train/val/test CSVs
|   `-- README.md
|-- docs/
|   |-- DSCI471 Project Proposal.md
|   |-- DSCI471 Final Project.md   # Rubric
|   |-- GRADING.md                 # Reproduction guide
|   |-- ARTIFACTS.md               # Artifact map
|   |-- presentation.pdf
|   `-- reports/
|       |-- final_report.md
|       |-- final_report.html
|       |-- final_report.pdf
|       |-- evaluation_results.csv
|       |-- figures/
|       `-- ablations/
|-- models/                        # Generated locally; see models/README.md
|-- notebooks/
|   |-- richardson_experiment/     # v1-to-v5 development notebooks
|   `-- samii_experiment/          # Final pipeline demos
|-- scripts/
|   |-- export_report_figures.py
|   |-- build_report_pdf.py
|   |-- build_presentation_pdf.py
|   `-- run_sample_pipeline.py
|-- src/
|   |-- prepare_data.py
|   |-- captions.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|   |-- metrics.py
|   |-- search.py
|   `-- baseline_keyword.py
|-- requirements.txt
`-- README.md
```

---

## Reproduction Guide

Run from the repository root.

### 1. Create Environment

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Data

Either place the Kaggle dataset under `data/raw/fashion-dataset/`, or run:

```powershell
python src/download_kaggle_data.py
```

Kaggle API credentials are required for automatic download.

### 3. Prepare Data

```powershell
python src/prepare_data.py
python src/prepare_data.py --check
```

Expected output: valid train/validation/test splits under `data/processed/`.

### 4. Train the Final Model

```powershell
python src/train.py
```

This creates:

- `models/v4_image_encoder.weights.h5`
- `models/embeddings/train_text.npy`
- `models/embeddings/val_text.npy`

Training takes about 40-45 minutes on CPU depending on hardware.

### 5. Evaluate

```powershell
python src/evaluate.py
```

This writes:

- `docs/reports/evaluation_results.csv`
- `models/test_image_embeddings.npy`

### 6. Regenerate Figures

```powershell
python scripts/export_report_figures.py
```

### 7. Build the Report PDF

```powershell
python scripts/build_report_pdf.py
```

---

## Quick Smoke Tests

Validate processed splits:

```powershell
python src/prepare_data.py --check
```

Run only the TF-IDF path:

```powershell
python src/evaluate.py --baseline-only
```

Run a small sample evaluation:

```powershell
python src/evaluate.py --sample 500
```

Run a small sample training/evaluation pipeline:

```powershell
python scripts/run_sample_pipeline.py
```

Internet/Wi-Fi is required the first time the dual-encoder path loads
`sentence-transformers/all-MiniLM-L6-v2` from Hugging Face. If the model is
already cached locally, the run can proceed offline.

---

## Important Grading Notes

For graders who do not want to retrain:

- The committed final metrics are in `docs/reports/evaluation_results.csv`.
- The validation ablation metrics are in `docs/reports/ablations/`.
- The final written report is `docs/reports/final_report.pdf`.
- If dual-encoder inference must run immediately, include or request `models/v4_image_encoder.weights.h5` and `models/test_image_embeddings.npy`.
- The first dual-encoder inference or training run also needs internet/Wi-Fi
  for Hugging Face MiniLM unless the Hugging Face cache is already present.

For full reproducibility:

- Download Kaggle data.
- Run `src/prepare_data.py`.
- Run `src/train.py`.
- Run `src/evaluate.py`.

See [`GRADING.md`](GRADING.md) and [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) for exact artifact expectations.

---

## Main Files

| File | Purpose |
|---|---|
| [`src/prepare_data.py`](src/prepare_data.py) | Data cleaning, caption creation, split generation |
| [`src/captions.py`](src/captions.py) | Caption and query-style generation |
| [`src/model.py`](src/model.py) | EfficientNetB0 image tower and contrastive model |
| [`src/train.py`](src/train.py) | Final v4 model training |
| [`src/evaluate.py`](src/evaluate.py) | Unified TF-IDF vs dual-encoder evaluation |
| [`src/metrics.py`](src/metrics.py) | Retrieval metrics |
| [`docs/reports/final_report.md`](docs/reports/final_report.md) | Full final report |
| [`docs/reports/evaluation_results.csv`](docs/reports/evaluation_results.csv) | Final test metrics |
| [`docs/reports/figures/`](docs/reports/figures/) | EDA, metrics, and qualitative figures |
| [`notebooks/richardson_experiment/`](notebooks/richardson_experiment/) | v1-to-v5 ablation notebooks |
| [`notebooks/samii_experiment/`](notebooks/samii_experiment/) | Final pipeline notebooks |

---

## Credits

Dataset: Param Aggarwal, *Fashion Product Images Dataset*, Kaggle.  
Course: DSCI471 Deep Learning.  
Team: Richardson Chhin and Samii Shabuse.
