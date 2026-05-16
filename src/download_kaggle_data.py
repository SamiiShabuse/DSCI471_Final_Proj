import kagglehub
from pathlib import Path
import shutil

DATASET = "paramaggarwal/fashion-product-images-dataset"

def main():
    path = kagglehub.dataset_download(DATASET)

    print("KaggleHub downloaded dataset to path:", path)

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for item in Path(path).iterdir():
        if item.is_file():
            shutil.move(str(item), raw_dir / item.name)

if __name__ == "__main__":
    main()
    