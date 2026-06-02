# Report Figures

Static figures for `final_report.md`, `final_report.pdf`, `README.md`, and `presentation_slides.md`.

## Regenerate

```powershell
python scripts/export_report_figures.py
```

Requires project dependencies (`requirements.txt`), processed data, and trained v4 weights (`models/v4_image_encoder.weights.h5`) for the retrieval-demo images.

## Files

| File | Description |
|---|---|
| `eda_article_distribution.png` | Top 20 article-type counts after preprocessing |
| `eda_color_distribution.png` | Top 15 color counts |
| `eda_master_category_distribution.png` | Master-category counts |
| `eda_text_length_distribution.png` | Distribution of `product_text` lengths |
| `eda_sample_images_grid.png` | Representative image grid across article types |
| `test_metrics_comparison.png` | 2x2 bar chart - Top-1, Top-5, MRR, P@5 on the test set |
| `shopper_metrics_comparison.png` | Shopper-query metrics where the dual-encoder wins Top-5 and MRR |
| `ablation_progression.png` | v1 to v4 validation Recall@1 progression |
| `v4_training_loss.png` | v4 contrastive training loss curve |
| `v4_vs_v5_per_style.png` | v4 vs v5 recall by query style on validation |
| `final_recall_comparison.png` | Final model recall comparison during development |
| `demo_success_templated.png` | Qualitative dual-encoder success case |
| `demo_head_to_head_brand.png` | TF-IDF vs dual-encoder brand-query comparison |
| `demo_failure_shopper.png` | Shopper-query failure case for the dual-encoder |
