"""Load retrieval models and run text → product search."""

import numpy as np
import pandas as pd
import tensorflow as tf
from sentence_transformers import SentenceTransformer

from baseline_keyword import build_keyword_index, search_keyword
from config import TEXT_MODEL_NAME
from model import (
    FrozenTextDualEncoder,
    build_image_encoder,
    encode_images,
    encode_texts,
    resolve_image_path,
)
from paths import PROJECT_ROOT, WEIGHTS_PATH

__all__ = [
    "build_keyword_index",
    "search_keyword",
    "load_dual_encoder",
    "search_dual_encoder",
    "encode_images",
    "encode_texts",
    "resolve_image_path",
]


def load_dual_encoder() -> tuple[FrozenTextDualEncoder, SentenceTransformer]:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Dual-encoder weights not found at {WEIGHTS_PATH}. "
            "Train first with: python src/train.py"
        )

    image_encoder, _ = build_image_encoder()
    model = FrozenTextDualEncoder(image_encoder)
    dummy = tf.zeros((1, 224, 224, 3))
    model.encode_images(dummy)
    image_encoder.load_weights(WEIGHTS_PATH)
    text_encoder = SentenceTransformer(TEXT_MODEL_NAME)
    return model, text_encoder


def search_dual_encoder(
    query: str,
    gallery_df: pd.DataFrame,
    image_embeddings: np.ndarray,
    text_encoder: SentenceTransformer,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return top-k gallery rows ranked by cosine similarity to the query."""
    query_embedding = encode_texts(text_encoder, [query])
    similarities = (query_embedding @ image_embeddings.T).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = gallery_df.iloc[top_indices].copy()
    results["similarity_score"] = similarities[top_indices]
    return results.reset_index(drop=True)
