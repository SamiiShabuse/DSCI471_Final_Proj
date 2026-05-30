---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-family: 'Segoe UI', Arial, sans-serif; }
  h1 { color: #1e40af; }
  h2 { color: #1e3a8a; border-bottom: 2px solid #dbeafe; padding-bottom: 0.2em; }
  blockquote { border-left: 4px solid #2563eb; background: #f8fafc; }
  table { font-size: 0.85em; }
  img { max-height: 420px; display: block; margin: 0 auto; }
---

# Multimodal Deep Learning Search Engine for E-commerce Fashion

**Richardson Chhin · Samii Shabuse · DSCI471**

May 2026

---

## Problem & motivation

- Shoppers describe products in natural language — e.g. *"a flowy floral dress with cap sleeves"* — but catalogs rely on keywords and metadata.
- We build a retriever that maps **text queries → product images** in a shared embedding space.
- Dataset: **44,265** fashion products (Kaggle Fashion Product Images).
- Better discovery → relevance, conversion, fewer mismatched clicks.

---

## Research question

> Can a dual-encoder deep learning model, trained on paired fashion images and text, **outperform traditional keyword-based e-commerce search**?

- **Baseline:** TF-IDF on `product_text` (caption + JSON description when available).
- **Proposed:** Dual-encoder — text query → embedding ← product image embedding.

---

## Dataset & preprocessing

| Item | Detail |
|---|---|
| Source | Kaggle Fashion Product Images (~44k) |
| Modalities | Image + CSV attributes + optional JSON description |
| Splits | 80/10/10 train / val / test (stratified by `articleType`) |
| Test gallery | **4,427** products |

Pipeline: `src/prepare_data.py` → captions + `product_text` per product.

---

## Query styles (evaluation)

Each test product generates a query from its own attributes:

| Style | Example |
|---|---|
| **templated** | `A men navy blue shirts, for casual wear in fall. Turtle Check Men Navy Blue Shirt.` |
| **shopper** | `men's navy blue shirt` |
| **brand** | `Turtle Check Men Navy Blue Shirt` |
| **short** | `navy shirt` |

Same four styles for TF-IDF and dual-encoder.

---

## Evaluation protocol

- **Gallery:** held-out test set (4,427 products).
- **Relevance:** each query targets **its own product ID** in the gallery.
- **Metrics:** Top-1, Top-5, MRR, Precision@5.
- **Development ablations** use validation Recall@K; **final table** uses test set.

---

## Architecture (final model v4)

```
Text query  →  MiniLM (frozen)  →  384-d embedding
                                        ↓ cosine similarity
Product image → EfficientNetB0 (fine-tuned) → 384-d embedding
```

- Contrastive loss (in-batch negatives), τ = 0.07.
- Selected after v1→v5 ablations — **pretrained text encoder = largest gain**.

---

## Baseline (TF-IDF)

- `TfidfVectorizer` on full product text index (50k features).
- Query and gallery in the **same text space** → strong on brand names and keyword overlap.
- Same test gallery and query styles as dual-encoder.

**Key asymmetry:** TF-IDF is text→text; dual-encoder is text→**image**.

---

## Training details

| Setting | Value |
|---|---|
| Image encoder | EfficientNetB0, ImageNet init, fine-tuned last 20 layers |
| Text encoder | `all-MiniLM-L6-v2` (frozen) |
| Loss | Symmetric contrastive (image↔text) |
| Final selection | **v4** beat v5 (rotation did not help) |

Richardson: ablation notebooks `richardson_experiment/01–08`. Samii: unified `src/` pipeline + evaluation.

---

## Ablation progression (validation)

![Ablation v1→v4 Recall@1](../reports/figures/ablation_progression.png)

Pretrained MiniLM text encoder (v4) nearly **doubled** Recall@1 over scratch text (v1).

---

## Results — test set (4,427 products)

![Test metrics comparison](../reports/figures/test_metrics_comparison.png)

**Answer:** TF-IDF wins **Top-1 on every query style**.

---

## Results — Top-1 accuracy

| Query type | TF-IDF | Dual-encoder |
|---|---|---|
| templated | **0.52** | 0.17 |
| shopper | **0.08** | 0.07 |
| brand | **0.65** | 0.14 |
| short | **0.07** | 0.05 |

Largest gaps: **brand** and **templated** (queries align with indexed text).

---

## Where dual-encoder wins (shopper)

![Shopper metrics](../reports/figures/shopper_metrics_comparison.png)

Dual-encoder: higher **Top-5** (0.24 vs 0.22) and **MRR** (0.167 vs 0.161) on natural shopper queries.

---

## Demo — success (templated query)

![Success example](../reports/figures/demo_success_templated.png)

Dual-encoder retrieves the correct **image** at rank 1.

---

## Demo — head-to-head (brand query)

![Head-to-head brand query](../reports/figures/demo_head_to_head_brand.png)

TF-IDF matches product names in text index; dual-encoder searches images.

---

## Demo — failure (shopper query)

![Failure example](../reports/figures/demo_failure_shopper.png)

Cross-modal matching is harder when attributes are ambiguous (many similar items).

---

## Limitations

1. **Modality gap** — text→image vs TF-IDF text→text (with rich JSON descriptions).
2. **Low Top-1 overall** — both models struggle on shopper/short queries (~5–8%).
3. **Synthetic queries** — generators ≠ real search logs.
4. **Compute** — ~40 min CPU training; no large hyperparameter search.

---

## Conclusion & future work

**Conclusion:** Dual-encoder improved through v1→v4 (MiniLM critical), but **keyword search wins Top-1** when the gallery is fully text-indexed. Cross-modal retrieval is viable and wins on some Top-5 / MRR metrics.

**Future work:** fine-tune text tower, hard negatives, CLIP-style domain pretraining, hybrid TF-IDF → dual reranking, real query logs.

---

## Reproducibility

```powershell
python src/prepare_data.py
python src/train.py
python src/evaluate.py
```

- Full report: `docs/reports/final_report.md` (+ PDF)
- Grading guide: `docs/GRADING.md`
- Results notebook: `notebooks/samii_experiment/04_final_results.ipynb`

---

## Team contributions

| Richardson Chhin | Samii Shabuse |
|---|---|
| v1→v5 experiment notebooks | Unified `src/` pipeline |
| Caption/query ablation design | TF-IDF + dual-encoder evaluation |
| Model selection (v4 final) | Data pipeline, docs, final report |

**Thank you!**
