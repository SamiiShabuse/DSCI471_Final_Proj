"""Shared hyperparameters and model constants."""

# Image encoder
IMG_SIZE = 224
TEXT_DIM = 384
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Contrastive training
TEMPERATURE = 0.07
BATCH_SIZE = 64
BASELINE_EPOCHS = 4
BASELINE_LR = 1e-3
FT_EPOCHS = 3
FT_LR = 1e-5
UNFREEZE_FROM = -20

# Encoding batch sizes
TEXT_ENCODE_BATCH_SIZE = 256
IMAGE_ENCODE_BATCH_SIZE = 64

# TF-IDF baseline
TFIDF_MAX_FEATURES = 50_000
