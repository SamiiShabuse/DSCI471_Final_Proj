"""Export static figures for final_report.md and presentation slides."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "docs" / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
ABLATIONS_DIR = REPORTS_DIR / "ablations"
EXPERIMENTS_DIR = PROJECT_ROOT / "models" / "experiments"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from paths import PROJECT_ROOT as ROOT  # noqa: E402
from paths import TEST_EMBEDDINGS_PATH, WEIGHTS_PATH  # noqa: E402


def _save(fig, name: str, dpi: int = 150) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")
    return path


def export_test_metrics_chart() -> None:
    results = pd.read_csv(REPORTS_DIR / "evaluation_results.csv")
    metrics = ["Top-1", "Top-5", "MRR", "Precision@5"]
    query_types = list(results["Query type"].unique())
    x = np.arange(len(query_types))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flatten(), metrics):
        baseline = results[results["Model"] == "TF-IDF"].set_index("Query type")[metric]
        dual = results[results["Model"].str.contains("Dual")].set_index("Query type")[metric]
        ax.bar(x - width / 2, [baseline.get(q, 0) for q in query_types], width, label="TF-IDF", color="#2563eb")
        ax.bar(x + width / 2, [dual.get(q, 0) for q in query_types], width, label="Dual-encoder (v4)", color="#64748b")
        ax.set_xticks(x)
        ax.set_xticklabels(query_types, rotation=15)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Retrieval performance on test set (4,427 products)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "test_metrics_comparison.png")


def export_ablation_chart() -> None:
    v134 = pd.read_csv(ABLATIONS_DIR / "v1_v2_v3_v4_comparison.csv")
    template_rows = v134[v134["Setup"].str.contains("template query", case=False)]
    labels = ["v1", "v2", "v3", "v4"]
    r1 = template_rows["R@1"].values

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#94a3b8", "#64748b", "#475569", "#1e293b"]
    ax.bar(labels, r1, color=colors)
    ax.set_ylabel("Recall@1 (validation)")
    ax.set_title("Ablation progression: templated query style", fontweight="bold")
    ax.set_ylim(0, max(r1) * 1.25)
    for i, v in enumerate(r1):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    _save(fig, "ablation_progression.png")


def export_shopper_comparison_chart() -> None:
    results = pd.read_csv(REPORTS_DIR / "evaluation_results.csv")
    shopper = results[results["Query type"] == "shopper"].set_index("Model")
    metrics = ["Top-1", "Top-5", "MRR"]
    tfidf = shopper.loc["TF-IDF", metrics].values
    dual = shopper.loc[shopper.index.str.contains("Dual"), metrics].iloc[0].values

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, tfidf, width, label="TF-IDF", color="#2563eb")
    ax.bar(x + width / 2, dual, width, label="Dual-encoder (v4)", color="#64748b")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Shopper queries: where dual-encoder is competitive", fontweight="bold")
    ax.set_ylim(0, max(max(tfidf), max(dual)) * 1.35)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for i, (t, d) in enumerate(zip(tfidf, dual)):
        winner = "Dual" if d > t else "TF-IDF"
        ax.text(i, max(t, d) + 0.008, f"↑ {winner}" if winner == "Dual" else "", ha="center", fontsize=9)
    plt.tight_layout()
    _save(fig, "shopper_metrics_comparison.png")


def copy_experiment_plots() -> None:
    copies = [
        ("v4_loss_curve.png", "v4_training_loss.png"),
        ("v4_vs_v5_per_style.png", "v4_vs_v5_per_style.png"),
        ("final_recall_comparison.png", "final_recall_comparison.png"),
    ]
    for src_name, dst_name in copies:
        src = EXPERIMENTS_DIR / src_name
        dst = FIGURES_DIR / dst_name
        if src.exists():
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  copied {dst.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  skip missing {src}")


def export_retrieval_demos() -> None:
    if not WEIGHTS_PATH.exists():
        print("  skip retrieval demos (train first: python src/train.py)")
        return

    from captions import QUERY_GENERATORS
    from config import TFIDF_MAX_FEATURES
    from model import encode_images, encode_texts, resolve_image_path
    from search import load_dual_encoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    test_df = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    model, text_encoder = load_dual_encoder()
    image_paths = [resolve_image_path(p, ROOT) for p in test_df["image_path"]]

    if TEST_EMBEDDINGS_PATH.exists() and np.load(TEST_EMBEDDINGS_PATH).shape[0] == len(test_df):
        image_emb = np.load(TEST_EMBEDDINGS_PATH)
        print("  loaded cached test embeddings")
    else:
        print("  encoding test gallery (may take several minutes)...")
        image_emb = encode_images(model, image_paths)
        np.save(TEST_EMBEDDINGS_PATH, image_emb)

    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = tfidf_vectorizer.fit_transform(test_df["product_text"].astype(str))

    def dual_top_k(query: str, k: int = 5) -> np.ndarray:
        q_emb = encode_texts(text_encoder, [query])
        sims = (q_emb @ image_emb.T).flatten()
        return sims.argsort()[::-1][:k]

    def save_retrieval_row(title: str, query: str, target_id: int, top_idx: np.ndarray, filename: str) -> None:
        fig, axes = plt.subplots(1, len(top_idx), figsize=(3 * len(top_idx), 3.4))
        if len(top_idx) == 1:
            axes = [axes]
        for ax, idx in zip(axes, top_idx):
            row = test_df.iloc[idx]
            img = Image.open(resolve_image_path(row["image_path"], ROOT))
            ax.imshow(img)
            ax.axis("off")
            mark = "\n(correct)" if row["id"] == target_id else ""
            ax.set_title(str(row["productDisplayName"])[:28] + mark, fontsize=8)
        rank = int(np.where(test_df.iloc[top_idx]["id"].values == target_id)[0][0]) + 1 if target_id in test_df.iloc[top_idx]["id"].values else None
        rank_label = f"rank {rank}" if rank else "not in top-5"
        fig.suptitle(f"{title}\nQuery: {query}\nCorrect product: {rank_label}", fontsize=10)
        plt.tight_layout()
        _save(fig, filename)

    def save_head_to_head(query: str, target_id: int, filename: str) -> None:
        q_emb = encode_texts(text_encoder, [query])
        dual_idx = (q_emb @ image_emb.T).flatten().argsort()[::-1][:5]
        q_vec = tfidf_vectorizer.transform([query])
        tfidf_idx = cosine_similarity(q_vec, tfidf_matrix).flatten().argsort()[::-1][:5]

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for col, idx in enumerate(tfidf_idx):
            row = test_df.iloc[idx]
            img = Image.open(resolve_image_path(row["image_path"], ROOT))
            axes[0, col].imshow(img)
            axes[0, col].axis("off")
            mark = "\n✓" if row["id"] == target_id else ""
            axes[0, col].set_title(str(row["productDisplayName"])[:24] + mark, fontsize=7)
        axes[0, 0].set_ylabel("TF-IDF", fontsize=11, fontweight="bold")

        for col, idx in enumerate(dual_idx):
            row = test_df.iloc[idx]
            img = Image.open(resolve_image_path(row["image_path"], ROOT))
            axes[1, col].imshow(img)
            axes[1, col].axis("off")
            mark = "\n✓" if row["id"] == target_id else ""
            axes[1, col].set_title(str(row["productDisplayName"])[:24] + mark, fontsize=7)
        axes[1, 0].set_ylabel("Dual-encoder", fontsize=11, fontweight="bold")
        fig.suptitle(f"Head-to-head: same brand query\n{query}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        _save(fig, filename)

    # Success: templated query rank 1
    success_row = None
    success_query = None
    for _, row in test_df.sample(800, random_state=77).iterrows():
        query = QUERY_GENERATORS["templated"](row)
        q_emb = encode_texts(text_encoder, [query])
        top1 = (q_emb @ image_emb.T).flatten().argsort()[::-1][0]
        if test_df.iloc[top1]["id"] == row["id"]:
            success_row, success_query = row, query
            break
    if success_row is not None:
        q_emb = encode_texts(text_encoder, [success_query])
        top_idx = (q_emb @ image_emb.T).flatten().argsort()[::-1][:5]
        save_retrieval_row("Success — dual-encoder (templated query)", success_query, success_row["id"], top_idx, "demo_success_templated.png")

    # Head-to-head brand query
    brand_row = test_df.sample(1, random_state=33).iloc[0]
    brand_query = QUERY_GENERATORS["brand"](brand_row)
    save_head_to_head(brand_query, brand_row["id"], "demo_head_to_head_brand.png")

    # Failure: shopper query not in top 5
    scan = test_df.sample(400, random_state=99)
    for _, row in scan.iterrows():
        query = QUERY_GENERATORS["shopper"](row)
        top_idx = dual_top_k(query, 5)
        if row["id"] not in test_df.iloc[top_idx]["id"].values:
            save_retrieval_row("Failure — shopper query (dual-encoder)", query, row["id"], top_idx, "demo_failure_shopper.png")
            break


def main() -> None:
    print("Exporting report figures...")
    export_test_metrics_chart()
    export_ablation_chart()
    export_shopper_comparison_chart()
    copy_experiment_plots()
    export_retrieval_demos()
    print("Done.")


if __name__ == "__main__":
    main()
