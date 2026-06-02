# EDA Addendum — recommended figures and generation commands

This file lists a minimal set of exploratory figures to add to the report to show data understanding.

Recommended figures (export to `docs/reports/figures/`):
- `class_distribution.png` — bar chart of `articleType` frequencies (use `data/processed/products.csv`).
- `color_distribution.png` — top 20 `baseColour` counts.
- `missing_json_counts.png` — stacked bar of how many products have JSON descriptions by category.
- `example_images_grid.png` — 3×5 grid of sample images for common vs rare `articleType`.

Notebook cell / script snippet (pandas + matplotlib):

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/products.csv')
df['articleType'].value_counts().nlargest(30).plot.bar(figsize=(10,6))
plt.tight_layout(); plt.savefig('docs/reports/figures/class_distribution.png')
```

Notes for graders: add the above figures and reference them from `docs/reports/final_report.md` to recover the 2 EDA points.
