import json
import re
from html import unescape
from pathlib import Path

import pandas as pd

_KEEP_PLURAL = {"jeans", "sunglasses", "pajamas", "shorts", "tights", "leggings"}


def safe(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def singularize(word: str) -> str:
    w = word.lower().strip()
    if w in _KEEP_PLURAL or not w.endswith("s"):
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("shes") or w.endswith("ches"):
        return w[:-2]
    if w.endswith("ss"):
        return w
    return w[:-1]


def build_caption(row) -> str:
    gender = safe(row.get("gender"))
    color = safe(row.get("baseColour"))
    article = safe(row.get("articleType"))
    usage = safe(row.get("usage"))
    season = safe(row.get("season"))
    name = safe(row.get("productDisplayName"))

    parts = []
    descriptor = " ".join(p for p in [gender, color, article] if p).lower()
    if descriptor:
        parts.append(f"A {descriptor}")

    context = []
    if usage:
        context.append(f"for {usage.lower()} wear")
    if season:
        context.append(f"in {season.lower()}")
    if context:
        parts.append(" ".join(context))

    caption = ", ".join(parts) + "." if parts else ""
    if name:
        caption = f"{caption} {name}.".strip()
    return caption


def build_product_text(row, description: str = "") -> str:
    caption = build_caption(row)
    description = description.strip()
    if description:
        return f"{caption} {description}".strip()
    return caption


def load_json_description(product_id: int, styles_dir: Path) -> str:
    path = styles_dir / f"{product_id}.json"
    if not path.exists():
        return ""

    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return ""

    raw = (
        payload.get("data", {})
        .get("productDescriptors", {})
        .get("description", {})
        .get("value", "")
    )
    if not raw:
        return ""

    text = re.sub(r"<[^>]+>", " ", raw)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text.upper() == "NA":
        return ""
    return text


def caption_templated(row) -> str:
    return build_caption(row)


def caption_short(row) -> str:
    color = safe(row.get("baseColour")).lower()
    article = singularize(safe(row.get("articleType")))
    return f"{color} {article}".strip()


def caption_shopper(row) -> str:
    gender = safe(row.get("gender")).lower()
    color = safe(row.get("baseColour")).lower()
    article = singularize(safe(row.get("articleType")))
    usage = safe(row.get("usage")).lower()
    gender_word = {
        "men": "men's",
        "women": "women's",
        "boys": "boys'",
        "girls": "girls'",
    }.get(gender, gender)
    if usage and usage != "casual":
        return f"{gender_word} {color} {usage} {article}".strip()
    return f"{gender_word} {color} {article}".strip()


def caption_brand(row) -> str:
    return safe(row.get("productDisplayName"))


QUERY_GENERATORS = {
    "templated": caption_templated,
    "short": caption_short,
    "shopper": caption_shopper,
    "brand": caption_brand,
}
