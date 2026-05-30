# Report figures

Static figures for `final_report.md`, `final_report.pdf`, and `presentation_slides.md`.

## Regenerate

```powershell
python scripts/export_report_figures.py
```

Requires project dependencies (`requirements.txt`) and trained v4 weights (`models/v4_image_encoder.weights.h5`).

## Files

| File | Description |
|---|---|
| `test_metrics_comparison.png` | 2×2 bar chart — Top-1, Top-5, MRR, P@5 on test set |
| `shopper_metrics_comparison.png` | Shopper query metrics (dual wins Top-5, MRR) |
| `ablation_progression.png` | v1→v4 validation Recall@1 (templated) |
| `v4_training_loss.png` | v4 contrastive training loss curve |
| `v4_vs_v5_per_style.png` | v4 vs v5 recall by query style (validation) |
| `final_recall_comparison.png` | Final model recall comparison (development) |
| `demo_success_templated.png` | Qualitative success — rank 1 |
| `demo_head_to_head_brand.png` | TF-IDF vs dual-encoder brand query |
| `demo_failure_shopper.png` | Shopper query failure case |
