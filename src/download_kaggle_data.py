import shutil
from pathlib import Path

import kagglehub

from paths import DATA_RAW_DIR

DATASET = "paramaggarwal/fashion-product-images-dataset"


def main():
    print(f"Downloading {DATASET}...")
    source = Path(kagglehub.dataset_download(DATASET))
    print("Downloaded to:", source)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        dest = DATA_RAW_DIR / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print(f"Dataset copied to {DATA_RAW_DIR.resolve()}")
    print("Next step: python src/prepare_data.py")


if __name__ == "__main__":
    main()
