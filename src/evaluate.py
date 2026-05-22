import pandas as pd

from baseline_keyword import build_keyword_index, search_keyword


def is_relevant(query_row, result_row):
    same_article = query_row["articleType"] == result_row["articleType"]
    same_gender = query_row["gender"] == result_row["gender"]
    same_color = query_row["baseColour"] == result_row["baseColour"]

    return same_article and same_gender and same_color


def precision_at_k(query_row, results_df, k=5):
    top_k_results = results_df.head(k)

    relevant_count = 0

    for _, result_row in top_k_results.iterrows():
        if is_relevant(query_row, result_row):
            relevant_count += 1

    return relevant_count / k


def reciprocal_rank(query_row, results_df):
    for rank, (_, result_row) in enumerate(results_df.iterrows(), start=1):
        if is_relevant(query_row, result_row):
            return 1 / rank

    return 0


def top_k_accuracy(query_row, results_df, k=5):
    top_k_results = results_df.head(k)

    for _, result_row in top_k_results.iterrows():
        if is_relevant(query_row, result_row):
            return 1

    return 0


def evaluate_keyword_baseline(sample_size=100, top_k=5):
    df, vectorizer, text_matrix = build_keyword_index()

    sample_df = df.sample(min(sample_size, len(df)), random_state=42)

    precision_scores = []
    top_k_scores = []
    reciprocal_ranks = []

    for _, query_row in sample_df.iterrows():
        query = query_row["product_text"]

        results = search_keyword(
            query=query,
            df=df,
            vectorizer=vectorizer,
            text_matrix=text_matrix,
            top_k=top_k,
        )
        results = results[results["id"] != query_row["id"]].head(top_k)

        precision_scores.append(precision_at_k(query_row, results, k=top_k))
        top_k_scores.append(top_k_accuracy(query_row, results, k=top_k))
        reciprocal_ranks.append(reciprocal_rank(query_row, results))

    metrics = {
        f"Precision@{top_k}": sum(precision_scores) / len(precision_scores),
        f"Top-{top_k} Accuracy": sum(top_k_scores) / len(top_k_scores),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }

    return metrics


if __name__ == "__main__":
    metrics = evaluate_keyword_baseline(sample_size=100, top_k=5)

    print("Keyword Baseline Evaluation")
    print("---------------------------")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")