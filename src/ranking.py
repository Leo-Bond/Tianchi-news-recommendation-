"""Feature engineering and ranking for candidate articles.

Primary ranker: LightGBM LambdaRank (grouped by user).
"""

from collections import Counter

import numpy as np
import pandas as pd
from .utils import safe_normalize


def build_user_interest_distribution(user_history, item_category, recent_n=20):
    distributions = {}
    for user_id, items in user_history.items():
        recent_items = items[-recent_n:]
        cats = [item_category.get(item) for item in recent_items if item_category.get(item) is not None]
        if not cats:
            distributions[user_id] = {}
            continue
        counts = Counter(cats)
        total = float(sum(counts.values()))
        distributions[user_id] = {cat: cnt / total for cat, cnt in counts.items()}
    return distributions


def build_user_embeddings(user_history, item_embeddings):
    user_emb = {}
    for user_id, items in user_history.items():
        vectors = [item_embeddings[item] for item in items if item in item_embeddings]
        if not vectors:
            continue
        user_emb[user_id] = safe_normalize(np.mean(vectors, axis=0))
    return user_emb


def build_feature_dataframe(
    candidates_by_user,
    user_history,
    item_embeddings,
    item_category,
    item_created_at,
    item_popularity,
    user_last_ts,
):
    user_emb = build_user_embeddings(user_history, item_embeddings)
    user_interest = build_user_interest_distribution(user_history, item_category)

    rows = []
    for user_id, candidates in candidates_by_user.items():
        history = user_history.get(user_id, [])
        last_item = history[-1] if history else None
        last_cat = item_category.get(last_item) if last_item is not None else None
        last_ts = user_last_ts.get(user_id, 0.0)
        u_vec = user_emb.get(user_id)
        interest_dist = user_interest.get(user_id, {})
        for article_id, recall_score in candidates:
            i_vec = item_embeddings.get(article_id)
            emb_sim = float(np.dot(u_vec, safe_normalize(i_vec))) if (u_vec is not None and i_vec is not None) else 0.0
            cat = item_category.get(article_id)
            cat_match = 1.0 if (last_cat is not None and cat == last_cat) else 0.0
            publish_gap = abs(float(last_ts) - float(item_created_at.get(article_id, last_ts)))
            pop = float(item_popularity.get(article_id, 0.0))
            interest_match = float(interest_dist.get(cat, 0.0)) if cat is not None else 0.0
            rows.append(
                {
                    "user_id": user_id,
                    "article_id": article_id,
                    "recall_score": float(recall_score),
                    "embedding_sim": emb_sim,
                    "category_match": cat_match,
                    "publish_time_gap": publish_gap,
                    "article_popularity": pop,
                    "user_interest_dist": interest_match,
                }
            )
    return pd.DataFrame(rows)


class GBDTLRRanker:
    """LightGBM LambdaRank ranker with recall-score fallback for degenerate labels."""

    feature_cols = [
        "recall_score",
        "embedding_sim",
        "category_match",
        "publish_time_gap",
        "article_popularity",
        "user_interest_dist",
    ]

    def __init__(self):
        self._available = False
        self._backend = "recall_score"
        self._lgb_ranker = None

        try:
            from lightgbm import LGBMRanker

            self._LGBMRanker = LGBMRanker
            self._available = True
        except Exception:
            raise ImportError("lightgbm is required for GBDTLRRanker")

    def fit(self, feature_df, label_df):
        merged = feature_df.merge(label_df, on=["user_id", "article_id"], how="left")
        merged = merged.sort_values(["user_id", "article_id"]).reset_index(drop=True)
        y = merged["label"].fillna(0).astype(int).values
        x = merged[self.feature_cols].fillna(0.0).values

        if np.unique(y).size > 1 and len(merged) > 0:
            groups = merged.groupby("user_id", sort=False).size().tolist()
            if groups and max(groups) > 1:
                self._lgb_ranker = self._LGBMRanker(
                    objective="lambdarank",
                    metric="ndcg",
                    ndcg_eval_at=[5],
                    learning_rate=0.05,
                    n_estimators=200,
                    num_leaves=31,
                    min_child_samples=20,
                    random_state=42,
                )
                self._lgb_ranker.fit(x, y, group=groups)
                self._backend = "lgbm_lambdarank"
        return self

    def predict(self, feature_df):
        x = feature_df[self.feature_cols].fillna(0.0).values
        if self._lgb_ranker is not None:
            pred = self._lgb_ranker.predict(x)
            out = feature_df[["user_id", "article_id"]].copy()
            out["rank_score"] = pred
            return out

        out = feature_df[["user_id", "article_id"]].copy()
        # When LambdaRank cannot be fit (e.g., single-class labels), keep recall ordering.
        out["rank_score"] = feature_df["recall_score"].astype(float).values
        return out
