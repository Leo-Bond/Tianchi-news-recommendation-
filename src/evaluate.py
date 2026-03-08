"""Evaluation metrics for the Tianchi news recommendation task.

The official competition metric is **MRR@5** (Mean Reciprocal Rank at 5):

    MRR@5 = mean over users of  1/rank(true_article)
            if true_article is in top-5 predictions, else 0.

Additional metrics provided: Hit Rate@k and NDCG@k.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


def reciprocal_rank(predictions: List[int], true_item: int, k: int = 5) -> float:
    """Compute the reciprocal rank of *true_item* in *predictions[:k]*."""
    for rank, item in enumerate(predictions[:k], start=1):
        if item == true_item:
            return 1.0 / rank
    return 0.0


def hit_at_k(predictions: List[int], true_item: int, k: int = 5) -> int:
    """Return 1 if *true_item* is in the top-*k* predictions, else 0."""
    return int(true_item in predictions[:k])


def ndcg_at_k(predictions: List[int], true_item: int, k: int = 5) -> float:
    """Compute NDCG@k for a single-relevant-item scenario."""
    for rank, item in enumerate(predictions[:k], start=1):
        if item == true_item:
            return 1.0 / np.log2(rank + 1)
    return 0.0


def evaluate(
    ranked_df: pd.DataFrame,
    label_df: pd.DataFrame,
    k: int = 5,
) -> Dict[str, float]:
    """Compute MRR@k, Hit@k and NDCG@k across all users.

    Parameters
    ----------
    ranked_df : pd.DataFrame
        Columns: [user_id, article_id, rank_score].  Already sorted
        descending by rank_score within each user.
    label_df : pd.DataFrame
        Columns: [user_id, click_article_id].  Ground-truth last click.
    k : int
        Cutoff.

    Returns
    -------
    dict with keys 'mrr', 'hit_rate', 'ndcg'.
    """
    ground_truth = label_df.set_index("user_id")["click_article_id"].to_dict()

    predictions = (
        ranked_df.sort_values("rank_score", ascending=False)
        .groupby("user_id")["article_id"]
        .apply(list)
        .to_dict()
    )

    mrr_scores, hit_scores, ndcg_scores = [], [], []
    for user_id, true_item in ground_truth.items():
        preds = predictions.get(user_id, [])
        mrr_scores.append(reciprocal_rank(preds, true_item, k))
        hit_scores.append(hit_at_k(preds, true_item, k))
        ndcg_scores.append(ndcg_at_k(preds, true_item, k))

    results = {
        f"mrr@{k}": float(np.mean(mrr_scores)),
        f"hit_rate@{k}": float(np.mean(hit_scores)),
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
    }
    logger.info("Evaluation results: %s", results)
    return results


def make_submission(ranked_df: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
    """Format predictions as the competition submission file.

    Parameters
    ----------
    ranked_df : pd.DataFrame
        Columns: [user_id, article_id, rank_score].
    topk : int
        Number of articles per user to include.

    Returns
    -------
    pd.DataFrame with columns [user_id, article_1, …, article_<topk>].
    """
    top = (
        ranked_df.sort_values("rank_score", ascending=False)
        .groupby("user_id")
        .head(topk)
    )
    top["rank"] = top.groupby("user_id").cumcount() + 1
    wide = top.pivot(index="user_id", columns="rank", values="article_id")
    wide.columns = [f"article_{i}" for i in wide.columns]
    wide.reset_index(inplace=True)
    return wide
