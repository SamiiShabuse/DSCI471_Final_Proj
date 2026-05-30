"""Train the v4 dual-encoder image tower (frozen MiniLM text encoder)."""

import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from sentence_transformers import SentenceTransformer

from config import (
    BASELINE_EPOCHS,
    BASELINE_LR,
    BATCH_SIZE,
    FT_EPOCHS,
    FT_LR,
    TEXT_MODEL_NAME,
    UNFREEZE_FROM,
)
from model import (
    FrozenTextDualEncoder,
    build_image_encoder,
    encode_texts,
    make_training_dataset,
    resolve_image_path,
)
from paths import DATA_PROCESSED_DIR, EMBEDDINGS_DIR, MODELS_DIR, PROJECT_ROOT, WEIGHTS_PATH
from prepare_data import validate_splits


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED_DIR / f"{name}.csv").reset_index(drop=True)


def encode_and_cache(text_encoder, captions, cache_path) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(captions):
            print(f"Loaded cached text embeddings: {cache_path.name}")
            return cached

    print(f"Encoding {len(captions):,} captions -> {cache_path.name}")
    embeddings = encode_texts(text_encoder, captions)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Train v4 dual-encoder image tower")
    parser.add_argument("--baseline-epochs", type=int, default=BASELINE_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FT_EPOCHS)
    parser.add_argument("--sample", type=int, default=0, help="Train on a subset for smoke tests")
    args = parser.parse_args()

    validate_splits()

    tf.random.set_seed(42)
    np.random.seed(42)

    train_df = load_split("train")
    val_df = load_split("val")
    if args.sample:
        train_df = train_df.sample(min(args.sample, len(train_df)), random_state=42)
        val_df = val_df.sample(min(max(args.sample // 5, 100), len(val_df)), random_state=42)
        print(f"Sample mode: train={len(train_df):,}, val={len(val_df):,}")

    text_encoder = SentenceTransformer(TEXT_MODEL_NAME)
    train_text = encode_and_cache(
        text_encoder, train_df["caption"], EMBEDDINGS_DIR / "train_text.npy"
    )
    val_text = encode_and_cache(
        text_encoder, val_df["caption"], EMBEDDINGS_DIR / "val_text.npy"
    )

    train_paths = [resolve_image_path(path, PROJECT_ROOT) for path in train_df["image_path"]]
    val_paths = [resolve_image_path(path, PROJECT_ROOT) for path in val_df["image_path"]]

    image_encoder, effnet = build_image_encoder()
    model = FrozenTextDualEncoder(image_encoder)
    model.compile(optimizer=tf.keras.optimizers.Adam(BASELINE_LR))

    train_ds = make_training_dataset(train_paths, train_text, BATCH_SIZE, training=True)
    val_ds = make_training_dataset(val_paths, val_text, BATCH_SIZE, training=False)

    print("\n=== Phase 1: frozen EfficientNet ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.baseline_epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=1,
                restore_best_weights=True,
            )
        ],
    )

    print("\n=== Phase 2: fine-tune last EfficientNet layers ===")
    effnet.trainable = True
    for layer in effnet.layers[:UNFREEZE_FROM]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR))
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=1,
                restore_best_weights=True,
            )
        ],
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    image_encoder.save_weights(WEIGHTS_PATH)
    print(f"\nSaved weights to {WEIGHTS_PATH}")
    print("Run evaluation with: python src/evaluate.py")


if __name__ == "__main__":
    main()
