"""Tests for feature engineering."""
import pandas as pd
import pytest

from src.feature_engineering import build_user_features, build_candidate_features


def _make_articles_df():
    return pd.DataFrame(
        {
            "article_id": [10, 20, 30, 40, 50],
            "category_id": [1, 1, 2, 2, 3],
            "words_count": [100, 200, 150, 250, 300],
        }
    )


def _make_click_df():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "click_article_id": [10, 20, 30, 40, 50],
            "click_timestamp": [1_000_000, 2_000_000, 3_000_000, 1_500_000, 2_500_000],
        }
    )


def test_build_user_features_columns():
    click_df = _make_click_df()
    articles_df = _make_articles_df()
    uf = build_user_features(click_df, articles_df)
    for col in ("click_count", "avg_words_count", "top_category_id", "last_click_ts"):
        assert col in uf.columns


def test_build_user_features_counts():
    click_df = _make_click_df()
    articles_df = _make_articles_df()
    uf = build_user_features(click_df, articles_df)
    assert uf.loc[1, "click_count"] == 3
    assert uf.loc[2, "click_count"] == 2


def test_build_candidate_features_basic():
    articles_df = _make_articles_df()
    click_df = _make_click_df()
    user_features = build_user_features(click_df, articles_df)
    click_history = {1: [10, 20, 30], 2: [40, 50]}
    recall_results = {1: [(40, 0.8), (50, 0.6)], 2: [(10, 0.7), (30, 0.5)]}

    feat_df = build_candidate_features(recall_results, user_features, click_history, articles_df)

    assert "recall_score" in feat_df.columns
    assert "in_user_history" in feat_df.columns
    # article 40 is NOT in user 1's history
    row = feat_df[(feat_df["user_id"] == 1) & (feat_df["article_id"] == 40)].iloc[0]
    assert row["in_user_history"] == 0
