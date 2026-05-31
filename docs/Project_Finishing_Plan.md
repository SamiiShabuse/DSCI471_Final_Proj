# Project Finishing Plan

**Project:** Multimodal Deep Learning Search Engine for E-commerce Fashion  
**Team:** Richardson Chhin, Samii Shabuse  
**Course:** DSCI471 Final Project

This document tracks the remaining work needed to deliver a complete submission that matches our [project proposal](DSCI471%20Project%20Proposal.md).

---

## Proposal Requirements → Deliverables

| Proposal section | Required deliverable |
|---|---|
| **Data Collection** | Full dataset pipeline: image–text pairs, preprocessing, train/val/test splits |
| **Methodology** | Dual-encoder (CNN + text encoder), contrastive training, cosine retrieval, **vs keyword baseline** |
| **Timeline Week 10** | Results, visualizations, **end-of-term presentation** |
| **Timeline Week 11** | **Final report**: methods, results, limitations |

---

## Current Status

### Done
- Raw Kaggle data present (~44k images + JSON)
- Full data pipeline (`src/prepare_data.py`, 44k splits with JSON descriptions)
- Keyword baseline + unified evaluation (`src/evaluate.py`)
- Final v4 model trained (`models/v4_image_encoder.weights.h5`)
- Comparison results (`docs/reports/evaluation_results.csv`)
- Consolidated repo docs (README, ARTIFACTS.md, GRADING.md, notebooks/)
- Phase 4 presentation outline + results notebook
- **Final report** (`docs/reports/final_report.md`, PDF, figures)
- **Presentation** (`docs/presentation.pdf`, `presentation_slides.md`)

### Gaps
- None required for submission. Optional: export Marp slides to PPTX via VS Code Marp extension.

---

## Phase 1 — Lock the Official Experiment

**Goal:** One canonical dataset, one text pipeline, one chosen final model.

- [x] **1.1 Choose canonical dataset and splits**
  - Use full ~44k product pipeline (80/10/10 train/val/test, stratified by `articleType`)
  - Output to `data/processed/`: `pairs.csv`, `train.csv`, `val.csv`, `test.csv`
  - Implemented in `src/prepare_data.py` (replaces 5k sample shortcut)

- [x] **1.2 Align text representation with proposal**
  - Combine CSV attributes + product name into structured captions (Richardson pipeline)
  - Append JSON `productDescriptors.description` when present and non-empty
  - Store both `caption` (attributes) and `product_text` (caption + optional description)

- [x] **1.3 Pick final model**
  - **Final model: v4** (`07_pretrained_text_encoder.ipynb`)
  - Architecture: EfficientNetB0 image encoder + frozen `sentence-transformers/all-MiniLM-L6-v2` text encoder + contrastive loss
  - Rationale: best R@1/R@5/R@10 on template queries; v5 (rotation) did not clearly beat v4

---

## Phase 2 — Unified Evaluation

**Goal:** Answer the research question with one fair comparison table.

- [x] **2.1 Single evaluation script/notebook**
  - `src/evaluate.py` — same held-out test gallery for TF-IDF and dual-encoder
  - Reports Top-1, Top-5, MRR, Precision@5 (per proposal)

- [x] **2.2 Query types**
  - templated, shopper, brand, short via `src/captions.py`

- [x] **2.3 Consistent relevance rule**
  - Retrieval-standard: each query targets its own product ID in the gallery

- [x] **2.4 Output**
  - Comparison table saved to `docs/reports/evaluation_results.csv`

**Run baseline now:**
```powershell
python src/evaluate.py --baseline-only
```

**Train dual-encoder then evaluate both:**
```powershell
python src/train.py
python src/evaluate.py
```

---

## Phase 3 — Consolidate the Repo

**Goal:** A grader/teammate can follow one clear path.

- [x] **3.1 Consolidated project structure**
  - Integrated Richardson's experiment notebooks into `notebooks/richardson_experiment/` (v1→v5 ablation story)
  - Ablation weights/plots in `models/experiments/`; metrics CSVs in `docs/reports/ablations/` only
  - Single pipeline: `src/` (CLI) + `notebooks/` (demos + experiments), shared via `src/paths.py`
  - Added `src/paths.py` as shared path/constants module
  - Added `src/__init__.py` package marker
  - Model weights excluded from git; `models/README.md` documents regeneration

- [x] **3.2 Notebooks integrated with src/**
  - `samii_experiment/01–04` — final pipeline demos
  - `richardson_experiment/01–08` — full v1→v5 development notebooks (see `richardson_experiment/README.md`)

- [x] **3.3 Reproducible artifacts documented**
  - `docs/ARTIFACTS.md`, `docs/reports/README.md`

- [x] **3.4 Requirements and README updated**
  - `requirements.txt`: direct dependencies with minimum versions
  - `README.md`, `docs/GRADING.md`, `data/README.md`: reproduction guide for graders
  - `prepare_data.py --check` validates split schema; sample evals no longer overwrite full results

---

## Phase 4 — Results, Visuals, and Demo

**Goal:** Week 10 presentation deliverables.

- [x] **4.1 Fill `04_final_results.ipynb`**
  - Comparison table + all metric bar charts
  - Evaluation protocol + nuanced shopper Top-5/MRR
  - Query-style examples
  - Ablation + training curves
  - Multi-query retrieval demos (4 styles)
  - Success example (templated rank-1)
  - TF-IDF vs dual-encoder head-to-head (brand query)
  - Shopper-query failure analysis

- [x] **4.2 Presentation slides**
  - Full outline in `docs/presentation.md` (title, 12 slides + appendices)

---

## Phase 5 — Final Report

**Goal:** Week 11 written report in `docs/reports/`.

- [x] **5.1 Write final report** (`docs/reports/final_report.md`)
  1. Introduction — research question, motivation ✓
  2. Related work — CLIP-style retrieval, fashion search ✓
  3. Data — source, size, image–text pairs, splits, preprocessing ✓
  4. Methods — encoders, contrastive loss, retrieval, baseline ✓
  5. Experiments — hyperparameters, v1–v5 ablation ✓
  6. Results — unified comparison table + figure references ✓
  7. Discussion — when deep learning wins vs loses ✓
  8. Limitations ✓
  9. Conclusion & future work ✓

- [x] **5.2 Explicitly answer the research question** with test-set numbers (Section 6.2)

---

## Priority Order (If Short on Time)

1. Unified eval table (baseline vs v4)
2. Fill `04_final_results.ipynb`
3. Presentation
4. Final report
5. Repo cleanup
6. JSON text enrichment (document as future work if skipped)

---

## Phase 1 Progress Log

| Date | Item | Notes |
|---|---|---|
| 2026-05-30 | Plan created | This document |
| 2026-05-30 | Structure consolidation | Integrated richardson_experiment/ + samii_experiment/, src/paths.py |
| 2026-05-30 | Phase 5 final report | `docs/reports/final_report.md` — full write-up + research question answer |
| 2026-05-30 | Report polish | Figures, PDF report, presentation PDF + Marp slides |
