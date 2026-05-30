# Multimodal Fashion Search — End-of-Term Presentation

**DSCI471 Final Project** · Richardson Chhin & Samii Shabuse

Export figures from `notebooks/samii_experiment/04_final_results.ipynb` or use plots in `models/experiments/`.

---

## Title slide

**Multimodal Deep Learning Search Engine for E-commerce Fashion**

Richardson Chhin · Samii Shabuse · DSCI471

---

## Slide 1 — Problem & motivation

- Shoppers describe products in natural language — e.g. *"a flowy floral dress with cap sleeves"* — but catalogs rely on keywords and metadata.
- We build a retriever that maps **text queries → product images** in a shared embedding space.
- Dataset: **44,265** fashion products (Kaggle Fashion Product Images).
- Better discovery → relevance, conversion, fewer mismatched clicks.

---

## Slide 2 — Research question

> Can a dual-encoder deep learning model, trained on paired fashion images and text, **outperform traditional keyword-based e-commerce search**?

- **Baseline:** TF-IDF on `product_text` (caption + JSON description when available).
- **Proposed:** Dual-encoder — text query → embedding ← product image embedding.

---

## Slide 3 — Dataset & preprocessing

| Item | Detail |
|---|---|
| Source | Kaggle Fashion Product Images (~44k) |
| Modalities | Image + CSV attributes + optional JSON description |
| Splits | 80/10/10 train / val / test (stratified by `articleType`) |
| Test gallery | **4,427** products |

Pipeline: `src/prepare_data.py` → captions + `product_text` per product.

---

## Slide 4 — Query styles (evaluation)

Each test product generates a query from its own attributes:

| Style | Example |
|---|---|
| **templated** | `A men navy blue shirts, for casual wear in fall. Turtle Check Men Navy Blue Shirt.` |
| **shopper** | `men's navy blue shirt` |
| **brand** | `Turtle Check Men Navy Blue Shirt` |
| **short** | `navy shirt` |

Same four styles for TF-IDF and dual-encoder.

---

## Slide 5 — Evaluation protocol

- **Gallery:** held-out test set (4,427 products).
- **Relevance:** each query targets **its own product ID** in the gallery.
- **Metrics (proposal):** Top-1, Top-5, MRR, Precision@5.
- **Development ablations** use validation Recall@K; **final table** uses test set.

---

## Slide 6 — Architecture (final model v4)

```
Text query  →  MiniLM (frozen)  →  384-d embedding
                                        ↓ cosine similarity
Product image → EfficientNetB0 (fine-tuned) → 384-d embedding
```

- Contrastive loss (in-batch negatives).
- Selected after v1→v5 ablations (pretrained text encoder = largest gain).

*Use a two-tower diagram in Slides.*

---

## Slide 7 — Baseline (TF-IDF)

- `TfidfVectorizer` on full product text index.
- Query and gallery in the **same text space** → strong on brand names and keyword overlap.
- Same test gallery and query styles as dual-encoder.

---

## Slide 8 — Training details

| Setting | Value |
|---|---|
| Image encoder | EfficientNetB0, ImageNet init, fine-tuned last 20 layers |
| Text encoder | `all-MiniLM-L6-v2` (frozen) |
| Loss | Contrastive (symmetric image↔text) |
| Final selection | **v4** beat v5 (rotation did not help) |

Richardson: ablation notebooks `richardson_experiment/01–08`. Samii: unified `src/` pipeline + evaluation.

---

## Slide 9 — Results (test set, 4,427 products)

### Top-1 accuracy

| Query type | TF-IDF | Dual-encoder |
|---|---|---|
| templated | **0.52** | 0.17 |
| shopper | **0.08** | 0.07 |
| brand | **0.65** | 0.14 |
| short | **0.07** | 0.05 |

**Answer:** TF-IDF wins Top-1 on every query style.

### Where dual-encoder is competitive (shopper)

| Metric | TF-IDF | Dual-encoder |
|---|---|---|
| Top-5 | 0.22 | **0.24** |
| MRR | 0.161 | **0.167** |

Dual-encoder often puts the correct product in the **top 5** for natural shopper queries even when rank 1 is wrong.

*Chart: 2×2 metric bars from `04_final_results.ipynb`.*

---

## Slide 10 — Demo (screenshots from notebook)

1. **Success** — templated query, dual-encoder rank 1 (correct image in top slot).
2. **Head-to-head** — same brand query: TF-IDF top row vs dual-encoder bottom row.
3. **Failure** — shopper query, correct product not in dual-encoder top 5.

Emphasize: dual-encoder retrieves **images**, not text snippets.

---

## Slide 11 — Limitations

1. **Modality gap** — text→image vs TF-IDF text→text (with rich JSON descriptions).
2. **Low Top-1 overall** — both models struggle on shopper/short queries (~5–8%).
3. **Synthetic queries** — templated/shopper generators ≠ real search logs.
4. **Compute** — ~40 min CPU training; embeddings cached for eval.

---

## Slide 12 — Conclusion & future work

**Conclusion:** Dual-encoder improved through v1→v4 (MiniLM text encoder critical), but **keyword search wins Top-1** when the gallery is fully text-indexed. Cross-modal retrieval is viable and wins on some Top-5 / MRR metrics.

**Future work:** fine-tune text tower on fashion data, hard negatives, CLIP-style domain pretraining, real query logs.

---

## Appendix A — Ablation (validation, templated query)

| v1 scratch | v2 captions | v3 rotation | **v4 MiniLM** |
|---|---|---|---|
| 0.11 | 0.08 | 0.07 | **0.21** |

Source: `docs/reports/ablations/v1_v2_v3_v4_comparison.csv`

---

## Appendix B — Reproducibility

```powershell
python src/prepare_data.py
python src/evaluate.py              # official results
python src/evaluate.py --sample 500 # smoke test only
```

Full guide: `docs/GRADING.md`

---

## Appendix C — Team contributions

| Richardson Chhin | Samii Shabuse |
|---|---|
| v1→v5 experiment notebooks | Unified `src/` pipeline |
| Caption/query ablation design | TF-IDF + dual-encoder evaluation |
| Model selection (v4 final) | Data pipeline, docs, Phase 4 notebook |
