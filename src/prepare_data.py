import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from captions import build_caption, build_product_text, load_json_description
from paths import DATA_PROCESSED_DIR, DATA_RAW_DIR

STYLES_DIR = DATA_RAW_DIR / "styles"
IMG_DIR = DATA_RAW_DIR / "images"

OUTPUT_COLUMNS = [
    "id",
    "image_path",
    "caption",
    "description",
    "product_text",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "usage",
    "productDisplayName",
]

SPLIT_COLUMNS = [
    "id",
    "image_path",
    "caption",
    "product_text",
    "articleType",
    "gender",
    "baseColour",
    "usage",
    "season",
    "productDisplayName",
]


def load_styles() -> pd.DataFrame:
    styles_path = DATA_RAW_DIR / "styles.csv"
    if not styles_path.exists():
        raise FileNotFoundError(f"Could not find {styles_path}")

    df = pd.read_csv(styles_path, on_bad_lines="skip")
    print(f"Loaded styles.csv rows: {len(df):,}")

    df["image_path"] = df["id"].apply(
        lambda product_id: str(IMG_DIR / f"{product_id}.jpg")
    )
    df = df[df["image_path"].apply(lambda path: Path(path).exists())]
    print(f"Rows with existing images: {len(df):,}")

    descriptions = [
        load_json_description(int(product_id), STYLES_DIR)
        for product_id in df["id"]
    ]
    df["description"] = descriptions

    df["caption"] = df.apply(build_caption, axis=1)
    df["product_text"] = df.apply(
        lambda row: build_product_text(row, row["description"]),
        axis=1,
    )

    with_description = (df["description"] != "").sum()
    print(f"Rows with JSON descriptions: {with_description:,}")

    type_counts = df["articleType"].value_counts()
    valid_types = type_counts[type_counts >= 10].index
    df = df[df["articleType"].isin(valid_types)].reset_index(drop=True)
    print(f"After dropping rare article types: {len(df):,}")

    return df[OUTPUT_COLUMNS]


def save_splits(df: pd.DataFrame) -> None:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["articleType"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["articleType"],
    )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df[SPLIT_COLUMNS].to_csv(DATA_PROCESSED_DIR / "pairs.csv", index=False)
    train_df[SPLIT_COLUMNS].to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    val_df[SPLIT_COLUMNS].to_csv(DATA_PROCESSED_DIR / "val.csv", index=False)
    test_df[SPLIT_COLUMNS].to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)

    products_out = df[
        [
            "id",
            "image_path",
            "product_text",
            "gender",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "usage",
            "productDisplayName",
        ]
    ]
    products_out.to_csv(DATA_PROCESSED_DIR / "products.csv", index=False)

    print(f"\nSaved processed files to {DATA_PROCESSED_DIR.resolve()}:")
    print(f"  pairs.csv : {len(df):,} products")
    print(f"  train.csv : {len(train_df):,}")
    print(f"  val.csv   : {len(val_df):,}")
    print(f"  test.csv  : {len(test_df):,}")
    print(f"  products.csv: {len(df):,}")


def validate_splits(processed_dir: Path | None = None) -> None:
    """Raise FileNotFoundError or ValueError if processed splits are missing or stale."""
    processed_dir = processed_dir or DATA_PROCESSED_DIR
    required = ("train.csv", "val.csv", "test.csv", "products.csv")
    missing_files = [name for name in required if not (processed_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing processed files in {processed_dir}: {', '.join(missing_files)}. "
            "Run: python src/prepare_data.py"
        )

    for name in ("train.csv", "val.csv", "test.csv"):
        path = processed_dir / name
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        missing_cols = [col for col in SPLIT_COLUMNS if col not in columns]
        if missing_cols:
            raise ValueError(
                f"{path.name} is missing columns {missing_cols}. "
                "Your splits may be from an old experiment copy. "
                "Re-run: python src/prepare_data.py"
            )

    products_cols = pd.read_csv(processed_dir / "products.csv", nrows=0).columns.tolist()
    if "product_text" not in products_cols:
        raise ValueError(
            "products.csv is missing product_text. Re-run: python src/prepare_data.py"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build train/val/test splits from Kaggle data")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing splits without rebuilding",
    )
    args = parser.parse_args()

    if args.check:
        validate_splits()
        print(f"OK: processed splits in {DATA_PROCESSED_DIR.resolve()} look valid.")
        return

    df = load_styles()
    save_splits(df)
    validate_splits()

    print("\nExample caption:")
    print(df["caption"].iloc[0])


if __name__ == "__main__":
    main()
