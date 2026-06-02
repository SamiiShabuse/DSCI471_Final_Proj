# Speaker notes and experiential reflection

Slide 1 — Title
- Quick one-line elevator pitch: multimodal dual-encoder for fashion product search.

Slide 2 — Problem & motivation
- Say: shoppers use natural language; catalogs rely on keywords; we aim to bridge that gap.

Slide 3 — Dataset & preprocessing
- Mention source (Kaggle Fashion Product Images) and final corpus size (44,265). Note split seed and stratification by `articleType`.

Slide 4 — Query styles
- Explain why four synthetic styles were chosen and that they mirror likely shopper phrasing.

Slide 5 — Evaluation protocol
- Emphasize fairness: identical gallery and target rule for TF-IDF and dual-encoder.

Slide 6 — Architecture
- Briefly explain: EfficientNetB0 for image tower (lightweight, good accuracy/compute tradeoff), MiniLM for text (compact sentence embeddings). Mention contrastive loss and temperature.

Slide 7 — Baseline
- Note TF-IDF indexes `product_text` = caption + JSON description; explain why that gives TF-IDF an advantage on Top-1.

Slide 8 — Training details
- Call out two-phase training (freeze, then unfreeze last layers) and that text encoder was frozen to reduce compute and overfitting risk.

Slide 9 — Results
- Narrate major result: TF-IDF wins Top-1 across query styles; dual-encoder is competitive on Top-5/MRR for shopper queries.

Slide 10 — Demo
- Mention the notebook used to capture screenshots and demos (`notebooks/samii_experiment/04_final_results.ipynb`).

Slide 11 — Limitations
- Be explicit: synthetic queries, modality gap, frozen text encoder, single-positive eval.

Slide 12 — Conclusion & future work
- Quick bullets: fine-tune text tower, hard negatives, hybrid reranking, evaluate on real search logs.

Experiential reflection (for in-class scoring):
- Roles: Richardson (experiment notebooks, captions, ablations), Samii (data pipeline, unified training/eval, documentation).
- Practical challenges: balancing compute with repeatability; deciding to freeze text encoder for stability; deriving templated captions that match JSON descriptions.
