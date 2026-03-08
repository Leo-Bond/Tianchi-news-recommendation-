"""Tests for recall strategies."""
import pytest

from src.recall import ItemCF, UserCF, BPR, merge_recall_results


def _sample_history():
    return {
        1: [10, 20, 30],
        2: [20, 30, 40],
        3: [10, 40, 50],
        4: [30, 50, 60],
    }


class TestItemCF:
    def test_fit_returns_self(self):
        model = ItemCF()
        result = model.fit(_sample_history())
        assert result is model

    def test_recall_excludes_history(self):
        model = ItemCF().fit(_sample_history())
        candidates = model.recall(1, topk=10)
        article_ids = {a for a, _ in candidates}
        history_set = set(_sample_history()[1])
        assert article_ids.isdisjoint(history_set)

    def test_recall_topk(self):
        model = ItemCF().fit(_sample_history())
        candidates = model.recall(1, topk=2)
        assert len(candidates) <= 2

    def test_recall_unknown_user(self):
        model = ItemCF().fit(_sample_history())
        assert model.recall(999) == []

    def test_recall_scores_descending(self):
        model = ItemCF().fit(_sample_history())
        candidates = model.recall(1, topk=10)
        scores = [s for _, s in candidates]
        assert scores == sorted(scores, reverse=True)


class TestUserCF:
    def test_fit_returns_self(self):
        model = UserCF()
        assert model.fit(_sample_history()) is model

    def test_recall_excludes_history(self):
        model = UserCF().fit(_sample_history())
        candidates = model.recall(1, topk=10)
        article_ids = {a for a, _ in candidates}
        assert article_ids.isdisjoint(set(_sample_history()[1]))

    def test_recall_unknown_user(self):
        model = UserCF().fit(_sample_history())
        assert model.recall(999) == []


class TestBPR:
    def test_fit_and_recall(self):
        model = BPR(n_factors=8, n_epochs=3).fit(_sample_history())
        candidates = model.recall(1, topk=5)
        assert isinstance(candidates, list)
        assert all(isinstance(a, int) and isinstance(s, float) for a, s in candidates)

    def test_recall_excludes_history(self):
        model = BPR(n_factors=8, n_epochs=3).fit(_sample_history())
        candidates = model.recall(2, topk=10)
        article_ids = {a for a, _ in candidates}
        assert article_ids.isdisjoint(set(_sample_history()[2]))

    def test_recall_unknown_user(self):
        model = BPR(n_factors=8, n_epochs=2).fit(_sample_history())
        assert model.recall(999) == []


class TestMergeRecallResults:
    def test_merge_combines_users(self):
        r1 = {1: [(10, 0.9), (20, 0.5)], 2: [(30, 0.8)]}
        r2 = {1: [(10, 0.7), (40, 0.4)], 3: [(50, 0.6)]}
        merged = merge_recall_results([r1, r2])
        assert 1 in merged and 2 in merged and 3 in merged

    def test_merge_scores_are_descending(self):
        r1 = {1: [(10, 0.9), (20, 0.5), (30, 0.3)]}
        r2 = {1: [(20, 0.8), (10, 0.2), (40, 0.1)]}
        merged = merge_recall_results([r1, r2])
        scores = [s for _, s in merged[1]]
        assert scores == sorted(scores, reverse=True)
