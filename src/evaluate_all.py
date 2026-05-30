import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))

from captions import QUERY_GENERATORS
from dual_encoder import (
    FrozenTextDualEncoder,
    build_image_encoder,
    encode_images,
    encode_texts,
    make_training_dataset,
    resolve_image_path,
)
from metrics import compute_retrieval_metrics, ranks_from_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "docs/reports"

QUERY_STYLES = ["templated", "short", "shopper", "brand"]
WEIGHTS_PATH = MODELS_DIR / "v4_image_encoder.weights.h5"
RESULTS_PATH = REPORTS_DIR / "evaluation_results.csv"


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python src/prepare_data.py")
    return pd.read_csv(path).reset_index(drop=True)


def build_queries(gallery_df: pd.DataFrame, query_style: str) -> tuple[list[str], np.ndarray]:
    generator = QUERY_GENERATORS[query_style]
    captions = []
    target_indices = []
    id_to_idx = {product_id: idx for idx, product_id in enumerate(gallery_df["id"].values)}

    for idx, row in gallery_df.iterrows():
        caption = generator(row)
        if not caption:
            continue
        captions.append(caption)
        target_indices.append(id_to_idx[row["id"]])

    return captions, np.array(target_indices, dtype=np.int32)


def evaluate_tfidf(gallery_df: pd.DataFrame, query_styles: list[str]) -> list[dict]:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    gallery_matrix = vectorizer.fit_transform(gallery_df["product_text"].astype(str))
    rows = []

    for style in query_styles:
        captions, target_indices = build_queries(gallery_df, style)
        query_matrix = vectorizer.transform(captions)
        similarities = cosine_similarity(query_matrix, gallery_matrix)
        ranks = ranks_from_similarity(similarities, target_indices)
        metrics = compute_retrieval_metrics(ranks)
        rows.append({"Model": "TF-IDF", "Query type": style, **metrics})
        print(f"TF-IDF | {style:9} | Top-1={metrics['Top-1']:.3f} Top-5={metrics['Top-5']:.3f} "
              f"MRR={metrics['MRR']:.3f} P@5={metrics['Precision@5']:.3f}")

    return rows


def load_dual_encoder() -> tuple[FrozenTextDualEncoder, SentenceTransformer]:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Dual-encoder weights not found at {WEIGHTS_PATH}. "
            "Train first with: python src/train_dual_encoder.py"
        )

    image_encoder, _ = build_image_encoder()
    model = FrozenTextDualEncoder(image_encoder)
    dummy = tf.zeros((1, 224, 224, 3))
    model.encode_images(dummy)
    image_encoder.load_weights(WEIGHTS_PATH)
    text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model, text_encoder


def evaluate_dual_encoder(
    gallery_df: pd.DataFrame,
    query_styles: list[str],
    image_embeddings: np.ndarray | None = None,
) -> list[dict]:
    model, text_encoder = load_dual_encoder()
    image_paths = [
        resolve_image_path(path, PROJECT_ROOT) for path in gallery_df["image_path"]
    ]

    if image_embeddings is None:
        cache_path = MODELS_DIR / "test_image_embeddings.npy"
        if cache_path.exists() and np.load(cache_path).shape[0] == len(gallery_df):
            image_embeddings = np.load(cache_path)
            print(f"Loaded cached image embeddings: {cache_path}")
        else:
            print("Encoding gallery images...")
            image_embeddings = encode_images(model, image_paths)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, image_embeddings)

    rows = []
    for style in query_styles:
        captions, target_indices = build_queries(gallery_df, style)
        text_embeddings = encode_texts(text_encoder, captions)
        similarities = text_embeddings @ image_embeddings.T
        ranks = ranks_from_similarity(similarities, target_indices)
        metrics = compute_retrieval_metrics(ranks)
        rows.append({"Model": "Dual-encoder (v4)", "Query type": style, **metrics})
        print(
            f"Dual-encoder | {style:9} | Top-1={metrics['Top-1']:.3f} Top-5={metrics['Top-5']:.3f} "
            f"MRR={metrics['MRR']:.3f} P@5={metrics['Precision@5']:.3f}"
        )

    return rows


def save_results(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)
    print(f"\nSaved comparison table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified retrieval evaluation")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="Held-out split used as retrieval gallery (default: test)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Evaluate TF-IDF baseline only",
    )
    parser.add_argument(
        "--dual-only",
        action="store_true",
        help="Evaluate dual-encoder only",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Optional subsample size for quick runs",
    )
    args = parser.parse_args()

    gallery_df = load_split(args.split)
    if args.sample and args.sample < len(gallery_df):
        gallery_df = gallery_df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"Using subsample of {len(gallery_df):,} products")

    print(f"Gallery: {args.split}.csv ({len(gallery_df):,} products)\n")

    rows = []
    if not args.dual_only:
        print("=== TF-IDF Baseline ===")
        rows.extend(evaluate_tfidf(gallery_df, QUERY_STYLES))
        print()

    if not args.baseline_only:
        print("=== Dual-Encoder (v4) ===")
        try:
            rows.extend(evaluate_dual_encoder(gallery_df, QUERY_STYLES))
        except FileNotFoundError as exc:
            print(exc)
            if not rows:
                raise SystemExit(1) from exc

    save_results(rows, RESULTS_PATH)


if __name__ == "__main__":
    main()
