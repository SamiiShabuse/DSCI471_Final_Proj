# Contributing

Thanks for improving the Multimodal Deep Learning Search Engine for E-commerce Fashion. This repository is a reproducible course project, so contributions should make the retrieval pipeline easier to run, review, and explain.

## Good Contributions

- Improve setup, reproduction, or grading instructions.
- Add focused fixes to preprocessing, captions, training, search, or metrics.
- Improve report figures or artifact documentation.
- Add small validation scripts that make results easier to verify.
- Keep experiment notes tied to a clear dataset split, model version, and metric.

## Local Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Core validation commands:

```powershell
python src/prepare_data.py --check
python src/evaluate.py --baseline-only
python scripts/run_sample_pipeline.py
```

Full dual-encoder training requires the Kaggle Fashion Product Images dataset and may require internet access for the first Hugging Face MiniLM download.

## Data and Model Guidelines

Do not commit Kaggle raw data, generated processed datasets, private credentials, local cache files, or machine-specific paths. Model weights should be committed only when they are intentionally part of a reproducible artifact and documented in the README or artifact guide.

## Pull Request Checklist

- The change is focused and explained.
- README, artifact docs, or report docs were updated if behavior changed.
- A relevant smoke test or reproduction command was run.
- No raw Kaggle data, credentials, or local paths are included.
- Metric changes identify the query style, split, and model version.