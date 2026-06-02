# EDA Addendum

The exploratory data analysis figures have been generated and integrated into
`docs/reports/final_report.md` and `README.md`.

## Generated Figures

| Figure | Purpose |
|---|---|
| `docs/reports/figures/eda_article_distribution.png` | Shows class imbalance across the top article types |
| `docs/reports/figures/eda_color_distribution.png` | Shows the most common base colors |
| `docs/reports/figures/eda_master_category_distribution.png` | Shows catalog scale by master category |
| `docs/reports/figures/eda_text_length_distribution.png` | Shows distribution of `product_text` lengths |
| `docs/reports/figures/eda_sample_images_grid.png` | Shows representative catalog images |

## Regenerate

```powershell
python scripts/export_report_figures.py
```

## Key Dataset Facts

- Final processed corpus: 44,265 products
- Train/validation/test split: 35,412 / 4,426 / 4,427
- Article types: 107
- Base colors: 46
- Products with loaded JSON descriptions: 44,136
- Median `product_text` length: 674 characters
