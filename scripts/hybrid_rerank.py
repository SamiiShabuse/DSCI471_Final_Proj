"""Hybrid retrieval: TF-IDF shortlist -> dual-encoder rerank.

Produces `docs/reports/hybrid_results.csv` with Top-1/Top-5/MRR for the hybrid pipeline.
This script is safe to run; if dual-encoder weights are missing it will run TF-IDF only
and write a short report.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baseline_keyword import build_keyword_index
from paths import DATA_PROCESSED_DIR


def evaluate_hybrid(sample_n=500, tfidf_topk=50, rerank_k=5, out_path=None):
    df, vectorizer, text_matrix = build_keyword_index()

    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")
    if sample_n:
        test_df = test_df.sample(min(sample_n, len(test_df)), random_state=42)

    # Try to load dual encoder lazily (avoid heavy imports at module import time)
    dual_ok = True
    try:
        from search import load_dual_encoder, search_dual_encoder

        model, text_encoder = load_dual_encoder()
    except Exception:
        model = None
        text_encoder = None
        search_dual_encoder = None
        dual_ok = False

    # Precompute gallery image embeddings if dual available
    if dual_ok:
        # build gallery df and embeddings via evaluate pipeline conventions
        gallery_df = pd.read_csv(DATA_PROCESSED_DIR / "products.csv")
        # try to load cached embeddings file if present
        embeddings_path = ROOT / "models" / "test_image_embeddings.npy"
        if embeddings_path.exists():
            image_embeddings = np.load(embeddings_path)
        else:
            # fallback: cannot rerank without embeddings
            dual_ok = False

    results = []

    for _, row in test_df.iterrows():
        q = row["caption"]
        # TF-IDF shortlist
        q_vec = vectorizer.transform([q])
        sims = (q_vec @ text_matrix.T).toarray().flatten()
        top_idx = sims.argsort()[::-1][:tfidf_topk]
        shortlist = df.iloc[top_idx].reset_index(drop=True)

        if dual_ok:
            # compute embeddings for shortlist
            shortlist_embs = image_embeddings[top_idx]
            reranked = search_dual_encoder(q, shortlist, shortlist_embs, text_encoder, top_k=rerank_k)
            # take reranked top-1
            top1 = int(reranked.iloc[0]["id"]) if not reranked.empty else None
            topk = list(reranked["id"].astype(int).tolist())
        else:
            top1 = int(shortlist.iloc[0]["id"]) if not shortlist.empty else None
            topk = list(shortlist["id"].astype(int).tolist()[:rerank_k])

        results.append({"query_id": int(row["id"]), "target_id": int(row["id"]), "top1": top1, "topk": topk})

    # Compute simple metrics
    total = len(results)
    top1_correct = sum(1 for r in results if r["top1"] == r["target_id"])
    top5_correct = sum(1 for r in results if r["target_id"] in r["topk"])  # topk size = rerank_k

    out = {
        "pipeline": "hybrid" if dual_ok else "tfidf_only",
        "n_queries": total,
        "Top-1": top1_correct / total,
        "Top-5": top5_correct / total,
    }

    out_path = Path(out_path or (ROOT / "docs" / "reports" / "hybrid_results.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(out).to_frame("value").to_csv(out_path)
    print(f"Wrote hybrid results to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--tfidf-topk", type=int, default=50)
    parser.add_argument("--rerank-k", type=int, default=5)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    evaluate_hybrid(sample_n=args.sample, tfidf_topk=args.tfidf_topk, rerank_k=args.rerank_k, out_path=args.out)


if __name__ == "__main__":
    main()
