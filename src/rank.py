"""LightGBM-based ranking model for the re-ranking stage.

The ranker takes the feature DataFrame produced by
:func:`src.feature_engineering.build_candidate_features` and learns a
pointwise binary classifier (clicked = 1 / not clicked = 0).  At inference
time the predicted probabilities are used as ranking scores.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGB_AVAILABLE = False

from .utils import get_logger, timer

logger = get_logger(__name__)

FEATURE_COLS: List[str] = [
    "recall_score",
    "candidate_category_id",
    "candidate_words_count",
    "in_user_history",
    "category_match_top",
    "category_match_ratio",
    "user_click_count",
]


class LGBRanker:
    """Pointwise LightGBM ranker.

    Parameters
    ----------
    params : dict, optional
        LightGBM training parameters.  Sensible defaults are provided.
    n_estimators : int
        Number of boosting rounds.
    """

    _default_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    def __init__(self, params: dict | None = None, n_estimators: int = 200):
        if not _LGB_AVAILABLE:
            raise ImportError("lightgbm is required for LGBRanker.  Install it with: pip install lightgbm")
        self.params = {**self._default_params, **(params or {})}
        self.n_estimators = n_estimators
        self.model: lgb.Booster | None = None

    @timer
    def fit(
        self,
        train_df: pd.DataFrame,
        label_col: str = "label",
        valid_df: pd.DataFrame | None = None,
    ) -> "LGBRanker":
        """Train the ranker.

        Parameters
        ----------
        train_df : pd.DataFrame
            Feature table with a *label_col* column (1 = positive, 0 = negative).
        label_col : str
            Name of the target column in *train_df*.
        valid_df : pd.DataFrame, optional
            Validation set (same schema as *train_df*).
        """
        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[label_col].values

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
        callbacks = [lgb.log_evaluation(period=50)]
        valid_sets = [dtrain]
        valid_names = ["train"]

        if valid_df is not None:
            X_valid = valid_df[FEATURE_COLS].values
            y_valid = valid_df[label_col].values
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
            valid_sets.append(dvalid)
            valid_names.append("valid")

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Return predicted click probabilities for each candidate row."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        X = feat_df[FEATURE_COLS].values
        return self.model.predict(X)

    @timer
    def rank(self, feat_df: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
        """Add a *rank_score* column and return the top-*k* per user.

        Parameters
        ----------
        feat_df : pd.DataFrame
            Candidate feature table (output of :func:`build_candidate_features`).
        topk : int
            Number of articles to return per user.

        Returns
        -------
        pd.DataFrame with columns [user_id, article_id, rank_score].
        """
        feat_df = feat_df.copy()
        feat_df["rank_score"] = self.predict(feat_df)
        result = (
            feat_df.sort_values("rank_score", ascending=False)
            .groupby("user_id")
            .head(topk)
            .reset_index(drop=True)
        )
        return result[["user_id", "article_id", "rank_score"]]


def build_training_samples(
    feat_df: pd.DataFrame,
    label_df: pd.DataFrame,
    neg_ratio: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Attach binary labels and down-sample negatives.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Candidate feature table.
    label_df : pd.DataFrame
        DataFrame with columns [user_id, click_article_id] representing the
        ground-truth last click.
    neg_ratio : int
        Number of negative samples per positive sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame – balanced training set with a *label* column.
    """
    pos_map = label_df.set_index("user_id")["click_article_id"].to_dict()

    def _label(row):
        return 1 if row["article_id"] == pos_map.get(row["user_id"], -1) else 0

    feat_df = feat_df.copy()
    feat_df["label"] = feat_df.apply(_label, axis=1)

    pos_df = feat_df[feat_df["label"] == 1]
    neg_df = feat_df[feat_df["label"] == 0].sample(
        n=min(len(pos_df) * neg_ratio, len(feat_df[feat_df["label"] == 0])),
        random_state=seed,
    )
    return pd.concat([pos_df, neg_df]).sample(frac=1, random_state=seed).reset_index(drop=True)
