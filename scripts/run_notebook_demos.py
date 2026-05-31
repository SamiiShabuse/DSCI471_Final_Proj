"""Run 04_final_results.ipynb demo cells (gallery + retrieval examples)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "docs" / "reports" / "figures" / "notebook_run"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from captions import QUERY_GENERATORS
from config import TFIDF_MAX_FEATURES
from model import encode_images, encode_texts, resolve_image_path
from paths import TEST_EMBEDDINGS_PATH, WEIGHTS_PATH
from search import load_dual_encoder


def dual_rank(query, target_id, test_df, image_emb, text_encoder):
    q_emb = encode_texts(text_encoder, [query])
    sims = (q_emb @ image_emb.T).flatten()
    order = sims.argsort()[::-1]
    ids = test_df.iloc[order]["id"].astype(int).values
    target_id = int(target_id)
    match = np.where(ids == target_id)[0]
    return int(match[0]) + 1 if len(match) else None


def save_retrieval(query, target_id, test_df, image_emb, text_encoder, path, title_prefix=""):
    q_emb = encode_texts(text_encoder, [query])
    sims = (q_emb @ image_emb.T).flatten()
    top_idx = sims.argsort()[::-1][:5]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for ax, idx in zip(axes, top_idx):
        row = test_df.iloc[idx]
        img = Image.open(resolve_image_path(row["image_path"], PROJECT_ROOT))
        ax.imshow(img)
        ax.axis("off")
        suffix = "\n(correct)" if int(row["id"]) == int(target_id) else ""
        ax.set_title(str(row["productDisplayName"])[:30] + suffix, fontsize=8)
    rank = dual_rank(query, target_id, test_df, image_emb, text_encoder)
    rank_label = f"rank {rank}" if rank else "not in top-5"
    fig.suptitle(f"{title_prefix}Query: {query}\nCorrect product: {rank_label}", fontsize=10)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return rank


def save_head_to_head(query, target_id, test_df, image_emb, text_encoder, tfidf_matrix, vectorizer, path):
    q_emb = encode_texts(text_encoder, [query])
    dual_idx = (q_emb @ image_emb.T).flatten().argsort()[::-1][:5]
    q_vec = vectorizer.transform([query])
    tfidf_idx = cosine_similarity(q_vec, tfidf_matrix).flatten().argsort()[::-1][:5]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for col, idx in enumerate(tfidf_idx):
        row = test_df.iloc[idx]
        img = Image.open(resolve_image_path(row["image_path"], PROJECT_ROOT))
        axes[0, col].imshow(img)
        axes[0, col].axis("off")
        mark = "\n(correct)" if int(row["id"]) == int(target_id) else ""
        axes[0, col].set_title(str(row["productDisplayName"])[:24] + mark, fontsize=7)
    axes[0, 0].set_ylabel("TF-IDF", fontsize=11)
    for col, idx in enumerate(dual_idx):
        row = test_df.iloc[idx]
        img = Image.open(resolve_image_path(row["image_path"], PROJECT_ROOT))
        axes[1, col].imshow(img)
        axes[1, col].axis("off")
        mark = "\n(correct)" if int(row["id"]) == int(target_id) else ""
        axes[1, col].set_title(str(row["productDisplayName"])[:24] + mark, fontsize=7)
    axes[1, 0].set_ylabel("Dual-encoder", fontsize=11)
    fig.suptitle(f"Head-to-head brand query\n{query}", fontsize=10)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not WEIGHTS_PATH.exists():
        print(f"ERROR: weights missing at {WEIGHTS_PATH}")
        sys.exit(1)

    results = pd.read_csv(PROJECT_ROOT / "docs/reports/evaluation_results.csv")
    test_df = pd.read_csv(PROJECT_ROOT / "data/processed/test.csv")

    print("Loading dual-encoder...")
    model, text_encoder = load_dual_encoder()
    image_paths = [resolve_image_path(p, PROJECT_ROOT) for p in test_df["image_path"]]
    if TEST_EMBEDDINGS_PATH.exists() and np.load(TEST_EMBEDDINGS_PATH).shape[0] == len(test_df):
        image_emb = np.load(TEST_EMBEDDINGS_PATH)
        print(f"Loaded cached embeddings: {TEST_EMBEDDINGS_PATH.name}")
    else:
        print("Encoding test gallery...")
        image_emb = encode_images(model, image_paths)
        TEST_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(TEST_EMBEDDINGS_PATH, image_emb)
        print(f"Saved {TEST_EMBEDDINGS_PATH.name}")

    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = tfidf_vectorizer.fit_transform(test_df["product_text"].astype(str))
    print(f"Gallery ready: {len(test_df):,} products\n")

    # Multi-query demo
    demo_rows = {
        "templated": test_df.sample(1, random_state=11).iloc[0],
        "shopper": test_df.sample(1, random_state=22).iloc[0],
        "brand": test_df.sample(1, random_state=33).iloc[0],
        "short": test_df.sample(1, random_state=44).iloc[0],
    }
    for style, row in demo_rows.items():
        query = QUERY_GENERATORS[style](row)
        print(f"=== {style.upper()} ===")
        print("Target:", row["productDisplayName"])
        rank = save_retrieval(
            query, row["id"], test_df, image_emb, text_encoder,
            OUT_DIR / f"demo_{style}.png",
        )
        print(f"Rank: {rank}\n")

    # Success example
    print("=== SUCCESS (templated, rank 1) ===")
    success_row = success_query = None
    for _, row in test_df.sample(2000, random_state=77).iterrows():
        query = QUERY_GENERATORS["templated"](row)
        if dual_rank(query, row["id"], test_df, image_emb, text_encoder) == 1:
            success_row, success_query = row, query
            break
    if success_row is not None:
        print("Target:", success_row["productDisplayName"])
        save_retrieval(
            success_query, success_row["id"], test_df, image_emb, text_encoder,
            OUT_DIR / "demo_success.png", title_prefix="SUCCESS - ",
        )
        print("Saved demo_success.png\n")
    else:
        print("No rank-1 hit in sample\n")

    # Head-to-head
    print("=== HEAD-TO-HEAD (brand) ===")
    brand_row = test_df.sample(1, random_state=33).iloc[0]
    brand_query = QUERY_GENERATORS["brand"](brand_row)
    print("Target:", brand_row["productDisplayName"])
    print("Query:", brand_query)
    save_head_to_head(
        brand_query, brand_row["id"], test_df, image_emb, text_encoder,
        tfidf_matrix, tfidf_vectorizer, OUT_DIR / "demo_head_to_head.png",
    )
    print("Saved demo_head_to_head.png\n")

    # Failures
    print("=== FAILURES (shopper) ===")
    scan = test_df.sample(400, random_state=99)
    failures = []
    for _, row in scan.iterrows():
        query = QUERY_GENERATORS["shopper"](row)
        q_emb = encode_texts(text_encoder, [query])
        sims = (q_emb @ image_emb.T).flatten()
        top5_idx = sims.argsort()[::-1][:5]
        top5_ids = set(test_df.iloc[top5_idx]["id"].astype(int).values)
        if int(row["id"]) not in top5_ids:
            failures.append((row, query))
    print(f"Failures in sample: {len(failures)} / {len(scan)}")
    dual_shopper = results.loc[
        (results["Model"].str.contains("Dual")) & (results["Query type"] == "shopper"), "Top-1"
    ].iloc[0]
    tfidf_shopper = results.loc[
        (results["Model"] == "TF-IDF") & (results["Query type"] == "shopper"), "Top-1"
    ].iloc[0]
    print(f"Official Top-1 - TF-IDF: {tfidf_shopper:.3f}, Dual: {dual_shopper:.3f}")
    for i, (row, query) in enumerate(failures[:3]):
        print(f"\n--- Failure {i + 1} ---")
        print("Query:", query)
        print("Target:", row["productDisplayName"])
        save_retrieval(
            query, row["id"], test_df, image_emb, text_encoder,
            OUT_DIR / f"demo_failure_{i + 1}.png",
        )
        print(f"Saved demo_failure_{i + 1}.png")

    print(f"\nAll figures saved to: {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
