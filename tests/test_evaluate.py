"""Tests for evaluation metrics."""
import pandas as pd
import pytest

from src.evaluate import reciprocal_rank, hit_at_k, ndcg_at_k, evaluate, make_submission


def test_reciprocal_rank_hit():
    assert reciprocal_rank([10, 20, 30], 20, k=5) == pytest.approx(0.5)


def test_reciprocal_rank_miss():
    assert reciprocal_rank([10, 20, 30], 99, k=5) == 0.0


def test_reciprocal_rank_beyond_k():
    assert reciprocal_rank([10, 20, 30, 40, 50, 60], 60, k=5) == 0.0


def test_hit_at_k():
    assert hit_at_k([10, 20, 30], 20, k=3) == 1
    assert hit_at_k([10, 20, 30], 99, k=3) == 0


def test_ndcg_at_k_first_position():
    import math
    score = ndcg_at_k([10, 20], 10, k=5)
    assert score == pytest.approx(1.0 / math.log2(2))


def test_evaluate_basic():
    ranked_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "article_id": [10, 20, 30, 40, 50],
            "rank_score": [0.9, 0.5, 0.3, 0.8, 0.2],
        }
    )
    label_df = pd.DataFrame({"user_id": [1, 2], "click_article_id": [10, 40]})
    metrics = evaluate(ranked_df, label_df, k=5)
    assert metrics["mrr@5"] == pytest.approx(1.0)
    assert metrics["hit_rate@5"] == pytest.approx(1.0)


def test_make_submission_shape():
    ranked_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1],
            "article_id": [10, 20, 30, 40, 50],
            "rank_score": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )
    sub = make_submission(ranked_df, topk=5)
    assert "user_id" in sub.columns
    assert "article_1" in sub.columns
    assert len(sub) == 1
