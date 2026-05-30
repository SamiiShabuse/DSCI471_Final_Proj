import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from captions import build_caption, build_product_text, load_json_description

RAW_DIR = Path("data/raw/fashion-dataset")
PROCESSED_DIR = Path("data/processed")
STYLES_DIR = RAW_DIR / "styles"
IMG_DIR = RAW_DIR / "images"

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
    styles_path = RAW_DIR / "styles.csv"
    if not styles_path.exists():
        raise FileNotFoundError(f"Could not find {styles_path}")

    df = pd.read_csv(styles_path, on_bad_lines="skip")
    print(f"Loaded styles.csv rows: {len(df):,}")

    df["image_path"] = df["id"].apply(
        lambda product_id: str(IMG_DIR / f"{product_id}.jpg")
    )
    df = df[df["image_path"].apply(lambda path: Path(path).exists())]
    print(f"Rows with existing images: {len(df):,}")

    descriptions = []
    for product_id in df["id"]:
        descriptions.append(load_json_description(int(product_id), STYLES_DIR))
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

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df[SPLIT_COLUMNS].to_csv(PROCESSED_DIR / "pairs.csv", index=False)
    train_df[SPLIT_COLUMNS].to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df[SPLIT_COLUMNS].to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df[SPLIT_COLUMNS].to_csv(PROCESSED_DIR / "test.csv", index=False)

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
    products_out.to_csv(PROCESSED_DIR / "products.csv", index=False)

    print(f"\nSaved processed files to {PROCESSED_DIR.resolve()}:")
    print(f"  pairs.csv : {len(df):,} products")
    print(f"  train.csv : {len(train_df):,}")
    print(f"  val.csv   : {len(val_df):,}")
    print(f"  test.csv  : {len(test_df):,}")
    print(f"  products.csv (legacy): {len(df):,}")


def main():
    df = load_styles()
    save_splits(df)

    print("\nExample caption:")
    print(df["caption"].iloc[0])
    print("\nExample product_text (caption + JSON description when available):")
    sample = df[df["description"] != ""].head(1)
    if not sample.empty:
        print(sample["product_text"].iloc[0][:500], "...")
    else:
        print(df["product_text"].iloc[0])


if __name__ == "__main__":
    main()
