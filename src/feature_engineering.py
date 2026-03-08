"""Feature engineering: extract user-level and article-level features.

Features produced
-----------------
User features:
  - click_count          : total number of historical clicks
  - active_days          : number of distinct days the user was active
  - avg_words_count      : average word count of clicked articles
  - top_category_id      : most frequently clicked category

Article features (for a candidate article given a user):
  - candidate_category_id      : category of the candidate article
  - candidate_words_count      : word count of the candidate article
  - time_since_last_click_s    : seconds since the user's most recent click
  - in_user_history            : 1 if the article was already clicked
  - category_match_ratio       : fraction of user's history in same category
  - recall_score               : score assigned by the recall stage
"""

import numpy as np
import pandas as pd

from .utils import get_logger, timer

logger = get_logger(__name__)

_ONE_DAY_MS = 86_400_000  # milliseconds in a day


@timer
def build_user_features(click_df: pd.DataFrame, articles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a user-level feature table.

    Parameters
    ----------
    click_df : pd.DataFrame
        Click log (sorted by user_id, click_timestamp).
    articles_df : pd.DataFrame
        Article meta-data with columns [article_id, category_id, words_count].

    Returns
    -------
    pd.DataFrame with index = user_id and one column per feature.
    """
    df = click_df.merge(
        articles_df[["article_id", "category_id", "words_count"]],
        left_on="click_article_id",
        right_on="article_id",
        how="left",
    )

    agg = df.groupby("user_id").agg(
        click_count=("click_article_id", "count"),
        active_days=("click_timestamp", lambda s: pd.to_datetime(s, unit="ms").dt.date.nunique()),
        avg_words_count=("words_count", "mean"),
        top_category_id=("category_id", lambda s: s.mode().iloc[0] if len(s) > 0 else -1),
        last_click_ts=("click_timestamp", "max"),
    )
    logger.info("Built user features: shape=%s", agg.shape)
    return agg


@timer
def build_candidate_features(
    recall_results: dict,
    user_features: pd.DataFrame,
    click_history: dict,
    articles_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a feature row for each (user, candidate_article) pair.

    Parameters
    ----------
    recall_results : dict
        {user_id -> [(article_id, recall_score), ...]}
    user_features : pd.DataFrame
        Output of :func:`build_user_features`.
    click_history : dict
        {user_id -> [article_id, ...]} (ordered by click time).
    articles_df : pd.DataFrame
        Article meta-data.

    Returns
    -------
    pd.DataFrame with columns:
        user_id, article_id, recall_score, + engineered features.
    """
    article_meta = articles_df.set_index("article_id")[
        ["category_id", "words_count"]
    ].to_dict(orient="index")

    rows = []
    for user_id, candidates in recall_results.items():
        u_feat = user_features.loc[user_id] if user_id in user_features.index else None
        history_set = set(click_history.get(user_id, []))
        history_list = click_history.get(user_id, [])

        for article_id, recall_score in candidates:
            meta = article_meta.get(article_id, {})
            cat = meta.get("category_id", -1)
            wc = meta.get("words_count", 0)

            if u_feat is not None:
                last_ts = u_feat["last_click_ts"]
                top_cat = u_feat["top_category_id"]
                click_count = u_feat["click_count"]
            else:
                last_ts = 0
                top_cat = -1
                click_count = 0

            # fraction of history articles in the same category
            if history_list and cat != -1:
                same_cat = sum(
                    1
                    for aid in history_list
                    if article_meta.get(aid, {}).get("category_id") == cat
                )
                category_match_ratio = same_cat / len(history_list)
            else:
                category_match_ratio = 0.0

            rows.append(
                {
                    "user_id": user_id,
                    "article_id": article_id,
                    "recall_score": recall_score,
                    "candidate_category_id": cat,
                    "candidate_words_count": wc,
                    "in_user_history": int(article_id in history_set),
                    "category_match_top": int(cat == top_cat),
                    "category_match_ratio": category_match_ratio,
                    "user_click_count": click_count,
                }
            )

    feat_df = pd.DataFrame(rows)
    logger.info("Built candidate features: shape=%s", feat_df.shape)
    return feat_df
