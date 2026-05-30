from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fashion-dataset"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
ABLATIONS_DIR = REPORTS_DIR / "ablations"

PRODUCTS_CSV = DATA_PROCESSED_DIR / "products.csv"
WEIGHTS_PATH = MODELS_DIR / "v4_image_encoder.weights.h5"
TEST_EMBEDDINGS_PATH = MODELS_DIR / "test_image_embeddings.npy"
EVAL_RESULTS_PATH = REPORTS_DIR / "evaluation_results.csv"

QUERY_STYLES = ("templated", "short", "shopper", "brand")
