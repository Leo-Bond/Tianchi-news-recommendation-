import os
import tempfile
import unittest

import pandas as pd

from src import main
from src.baseline_itemcf import build_baseline_submission


class PipelineSplitTest(unittest.TestCase):
    def test_main_uses_multi_recall_ranking_module(self):
        self.assertEqual(main.build_baseline_submission.__module__, "src.multi_recall_ranking")

    def test_itemcf_only_baseline_runs(self):
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
            train = pd.DataFrame(
                [
                    {"user_id": 1, "click_article_id": 101, "click_timestamp": 1},
                    {"user_id": 1, "click_article_id": 102, "click_timestamp": 2},
                    {"user_id": 2, "click_article_id": 101, "click_timestamp": 1},
                    {"user_id": 2, "click_article_id": 103, "click_timestamp": 2},
                ]
            )
            test = pd.DataFrame(
                [
                    {"user_id": 1, "click_article_id": 102, "click_timestamp": 3},
                    {"user_id": 3, "click_article_id": 101, "click_timestamp": 1},
                ]
            )
            train.to_csv(os.path.join(data_dir, "train_click_log.csv"), index=False)
            test.to_csv(os.path.join(data_dir, "testA_click_log.csv"), index=False)

            path = build_baseline_submission(
                data_dir=data_dir,
                output_dir=out_dir,
                topk_recall=5,
                topk_submit=2,
                topk_sim=20,
                popular_fill_k=50,
            )
            self.assertTrue(os.path.exists(path))
            sub = pd.read_csv(path)
            self.assertEqual(list(sub.columns), ["user_id", "article_1", "article_2"])
            self.assertEqual(len(sub), 2)


if __name__ == "__main__":
    unittest.main()
