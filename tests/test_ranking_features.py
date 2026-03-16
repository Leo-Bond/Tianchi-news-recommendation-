import unittest

import numpy as np
import pandas as pd

from src.ranking import GBDTLRRanker, build_feature_dataframe


class RankingFeatureTest(unittest.TestCase):
    def test_build_feature_dataframe_contains_required_columns(self):
        candidates = {1: [(101, 0.9), (102, 0.6)]}
        user_history = {1: [100]}
        item_embeddings = {
            100: np.array([1.0, 0.0]),
            101: np.array([0.9, 0.1]),
            102: np.array([0.1, 0.9]),
        }
        item_category = {100: 1, 101: 1, 102: 2}
        item_created_at = {101: 1000.0, 102: 990.0}
        item_popularity = {101: 1.0, 102: 0.5}
        user_last_ts = {1: 995.0}

        df = build_feature_dataframe(
            candidates_by_user=candidates,
            user_history=user_history,
            item_embeddings=item_embeddings,
            item_category=item_category,
            item_created_at=item_created_at,
            item_popularity=item_popularity,
            user_last_ts=user_last_ts,
        )
        self.assertEqual(set(df.columns), {
            "user_id",
            "article_id",
            "recall_score",
            "embedding_sim",
            "category_match",
            "publish_time_gap",
            "article_popularity",
            "user_interest_dist",
        })
        self.assertEqual(len(df), 2)

    def test_ranker_predict_returns_rank_score(self):
        feature_df = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "article_id": 101,
                    "recall_score": 0.9,
                    "embedding_sim": 0.8,
                    "category_match": 1.0,
                    "publish_time_gap": 10.0,
                    "article_popularity": 1.0,
                    "user_interest_dist": 0.7,
                },
                {
                    "user_id": 1,
                    "article_id": 102,
                    "recall_score": 0.5,
                    "embedding_sim": 0.4,
                    "category_match": 0.0,
                    "publish_time_gap": 100.0,
                    "article_popularity": 0.5,
                    "user_interest_dist": 0.2,
                },
            ]
        )
        labels = pd.DataFrame([{"user_id": 1, "article_id": 101, "label": 1}])

        ranker = GBDTLRRanker().fit(feature_df, labels)
        out = ranker.predict(feature_df)
        self.assertIn("rank_score", out.columns)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
