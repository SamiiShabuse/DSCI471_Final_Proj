"""Dual-encoder architecture, image loading, and embedding helpers."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

from config import (
    IMAGE_ENCODE_BATCH_SIZE,
    IMG_SIZE,
    TEMPERATURE,
    TEXT_DIM,
    TEXT_ENCODE_BATCH_SIZE,
)


def resolve_image_path(path: str, project_root: Path | None = None) -> str:
    raw = Path(path)
    if raw.is_absolute() and raw.exists():
        return str(raw)

    root = project_root or Path(".")
    candidates = [
        root / path,
        root / "data/raw/fashion-dataset/images" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Image not found for path: {path}")


def build_image_encoder(out_dim: int = TEXT_DIM):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        pooling="avg",
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    features = base(inputs, training=False)
    features = layers.Dense(512, activation="gelu")(features)
    features = layers.Dropout(0.1)(features)
    outputs = layers.Dense(out_dim)(features)
    return Model(inputs, outputs, name="image_encoder"), base


class FrozenTextDualEncoder(Model):
    def __init__(self, image_encoder, temperature: float = TEMPERATURE):
        super().__init__()
        self.img_enc = image_encoder
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def encode_images(self, images, training=False):
        embeddings = self.img_enc(images, training=training)
        return tf.math.l2_normalize(embeddings, axis=-1)

    def contrastive_loss(self, image_embeddings, text_embeddings):
        logits = tf.matmul(image_embeddings, text_embeddings, transpose_b=True) / self.temperature
        labels = tf.range(tf.shape(logits)[0])
        loss_i = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        loss_t = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.transpose(logits), from_logits=True
        )
        return tf.reduce_mean((loss_i + loss_t) / 2.0)

    def train_step(self, data):
        images, text_embeddings = data
        with tf.GradientTape() as tape:
            image_embeddings = self.encode_images(images, training=True)
            loss = self.contrastive_loss(image_embeddings, text_embeddings)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        images, text_embeddings = data
        image_embeddings = self.encode_images(images, training=False)
        loss = self.contrastive_loss(image_embeddings, text_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def load_image(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    image = tf.io.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return tf.cast(image, tf.float32)


def make_image_dataset(image_paths, batch_size: int = IMAGE_ENCODE_BATCH_SIZE, training: bool = False):
    dataset = tf.data.Dataset.from_tensor_slices(list(image_paths))
    if training:
        dataset = dataset.shuffle(min(len(image_paths), 8192), reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=training)
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_training_dataset(
    image_paths,
    text_embeddings,
    batch_size: int = IMAGE_ENCODE_BATCH_SIZE,
    training: bool = False,
):
    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), text_embeddings))
    if training:
        dataset = dataset.shuffle(min(len(image_paths), 8192), reshuffle_each_iteration=True)

    def _load(path, text_embedding):
        return load_image(path), text_embedding

    dataset = dataset.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=training)
    return dataset.prefetch(tf.data.AUTOTUNE)


def encode_images(
    model: FrozenTextDualEncoder,
    image_paths,
    batch_size: int = IMAGE_ENCODE_BATCH_SIZE,
) -> np.ndarray:
    dataset = make_image_dataset(image_paths, batch_size=batch_size, training=False)
    chunks = []
    for images in dataset:
        embeddings = model.encode_images(images, training=False)
        chunks.append(embeddings.numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def encode_texts(text_encoder, captions, batch_size: int = TEXT_ENCODE_BATCH_SIZE) -> np.ndarray:
    return text_encoder.encode(
        list(captions),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
