import numpy as np


def ranks_from_similarity(similarities: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    """Return 1-based rank of each target in the similarity matrix (0 if missing)."""
    ranked = np.argsort(-similarities, axis=1)
    ranks = np.zeros(len(target_indices), dtype=np.int32)
    for i, target in enumerate(target_indices):
        matches = np.where(ranked[i] == target)[0]
        ranks[i] = int(matches[0]) + 1 if len(matches) else 0
    return ranks


def compute_retrieval_metrics(
    ranks: np.ndarray,
    top_k_values=(1, 5),
    precision_k: int = 5,
) -> dict[str, float]:
    n = len(ranks)
    if n == 0:
        return {}

    metrics = {}
    for k in top_k_values:
        metrics[f"Top-{k}"] = float(np.mean((ranks > 0) & (ranks <= k)))

    reciprocal = np.where(ranks > 0, 1.0 / ranks, 0.0)
    metrics["MRR"] = float(np.mean(reciprocal))

    precision = np.where((ranks > 0) & (ranks <= precision_k), 1.0 / precision_k, 0.0)
    metrics[f"Precision@{precision_k}"] = float(np.mean(precision))

    return metrics
