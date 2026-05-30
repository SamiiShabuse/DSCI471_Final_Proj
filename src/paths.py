from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fashion-dataset"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"  # ablation weights, plots, pickles
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
ABLATIONS_DIR = REPORTS_DIR / "ablations"  # ablation metrics CSVs (report tables)

PRODUCTS_CSV = DATA_PROCESSED_DIR / "products.csv"
WEIGHTS_PATH = MODELS_DIR / "v4_image_encoder.weights.h5"
TEST_EMBEDDINGS_PATH = MODELS_DIR / "test_image_embeddings.npy"
EVAL_RESULTS_PATH = REPORTS_DIR / "evaluation_results.csv"
EVAL_RESULTS_SAMPLE_PATH = REPORTS_DIR / "evaluation_results_sample.csv"

QUERY_STYLES = ("templated", "short", "shopper", "brand")
