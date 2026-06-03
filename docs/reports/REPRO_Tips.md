# Quick reproduction tips (for graders)

1) Quick smoke (no heavy compute)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Run a sample train (small) and evaluation (writes a sample CSV)
python scripts/run_sample_pipeline.py
```

2) Hybrid reranker (uses TF-IDF shortlist and cached embeddings if present)

```powershell
# If you have models/test_image_embeddings.npy and model weights, hybrid rerank will rerank.
python scripts/hybrid_rerank.py --sample 500 --tfidf-topk 50 --rerank-k 5
```

3) If you cannot download the Kaggle dataset, you can still verify TF-IDF baseline code paths by placing a small subset CSV at `data/processed/products.csv` formatted like the project splits (columns: id, product_text, image_path, etc.) and then run:

```powershell
python src/evaluate.py --baseline-only --sample 100
```

4) Where results are saved
- `docs/reports/evaluation_results.csv` - committed final results.
- `docs/reports/evaluation_results_sample.csv` - sample results created by smoke runs.
- `docs/reports/hybrid_results.csv` - hybrid reranker output (if run).
