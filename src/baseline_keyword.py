import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from paths import PRODUCTS_CSV


def build_keyword_index(csv_path=PRODUCTS_CSV):
    df = pd.read_csv(csv_path)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    text_matrix = vectorizer.fit_transform(df["product_text"])

    return df, vectorizer, text_matrix


def search_keyword(query, df, vectorizer, text_matrix, top_k=5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, text_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["similarity_score"] = similarities[top_indices]

    return results[
        [
            "id",
            "productDisplayName",
            "product_text",
            "image_path",
            "gender",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "usage",
            "similarity_score",
        ]
    ]


if __name__ == "__main__":
    df, vectorizer, text_matrix = build_keyword_index()
    query = "blue casual men's shirt"
    results = search_keyword(query, df, vectorizer, text_matrix, 5)
    print(results[["id", "productDisplayName", "similarity_score"]])
