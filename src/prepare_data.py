from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/fashion-dataset")
PROCESSED_DIR = Path("data/processed")


def build_product_text(row):
    fields = [
        row.get("gender", ""),
        row.get("masterCategory", ""),
        row.get("subCategory", ""),
        row.get("articleType", ""),
        row.get("baseColour", ""),
        row.get("season", ""),
        row.get("usage", ""),
        row.get("productDisplayName", ""),
    ]

    return " ".join(str(field) for field in fields if pd.notna(field))


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    styles_path = RAW_DIR / "styles.csv"

    if not styles_path.exists():
        raise FileNotFoundError("Could not find data/raw/styles.csv")

    df = pd.read_csv(styles_path, on_bad_lines="skip")

    print("Original rows:", len(df))
    print("Columns:", df.columns.tolist())

    df["image_path"] = df["id"].apply(lambda product_id: str(RAW_DIR / "images" / f"{product_id}.jpg"))

    df = df[df["image_path"].apply(lambda path: Path(path).exists())]

    print("Rows with existing images:", len(df))

    df["product_text"] = df.apply(build_product_text, axis=1)

    useful_columns = [
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

    df = df[useful_columns]

    # Use smaller sample first so your computer/Colab does not explode.
    df = df.sample(min(len(df), 5000), random_state=42)

    output_path = PROCESSED_DIR / "products.csv"
    df.to_csv(output_path, index=False)

    print("Saved cleaned dataset to:", output_path)
    print("Final rows:", len(df))
    print("\nExample product text:")
    print(df["product_text"].iloc[0])


if __name__ == "__main__":
    main()