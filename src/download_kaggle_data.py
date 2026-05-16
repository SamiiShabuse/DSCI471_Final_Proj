import kagglehub
from pathlib import Path
import shutil

DATASET = "paramaggarwal/fashion-product-images-dataset"

def main():
    path = kagglehub.dataset_download(DATASET)

    print("KaggleHub downloaded dataset to path:", path)

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()
    