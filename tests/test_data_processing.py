"""Tests for data processing utilities."""
import pandas as pd
import pytest

from src.data_processing import build_user_click_history, split_last_click


def _make_click_df():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "click_article_id": [10, 20, 30, 40, 50],
            "click_timestamp": [100, 200, 300, 150, 250],
        }
    )


def test_build_user_click_history():
    df = _make_click_df()
    history = build_user_click_history(df)
    assert set(history.keys()) == {1, 2}
    assert history[1] == [10, 20, 30]
    assert history[2] == [40, 50]


def test_split_last_click_label():
    df = _make_click_df()
    hist_df, label_df = split_last_click(df)
    # Each user should have exactly one label row
    assert len(label_df) == 2
    # Label rows should be the last (highest timestamp) click
    labels = label_df.set_index("user_id")["click_article_id"].to_dict()
    assert labels[1] == 30
    assert labels[2] == 50


def test_split_last_click_no_overlap():
    df = _make_click_df()
    hist_df, label_df = split_last_click(df)
    # The two splits together should account for all original rows
    assert len(hist_df) + len(label_df) == len(df)


def test_split_last_click_single_click_user():
    df = pd.DataFrame(
        {
            "user_id": [99],
            "click_article_id": [7],
            "click_timestamp": [1000],
        }
    )
    hist_df, label_df = split_last_click(df)
    assert len(label_df) == 1
    assert len(hist_df) == 0
