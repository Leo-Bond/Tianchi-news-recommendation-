import unittest

import numpy as np

from src.recall import YouTubeDNNRecall


class YouTubeDNNRecallTest(unittest.TestCase):
    def setUp(self):
        self.history = {
            1: [101, 102],
            2: [101, 103],
        }
        self.embeddings = {
            101: np.array([1.0, 0.0]),
            102: np.array([0.9, 0.1]),
            103: np.array([0.0, 1.0]),
            104: np.array([0.8, 0.2]),
        }

    def test_proxy_backend_recall(self):
        model = YouTubeDNNRecall(use_deepmatch=False).fit(self.history, self.embeddings)
        recalled = model.recall(1, topk=3)
        self.assertGreaterEqual(len(recalled), 1)
        self.assertEqual(recalled[0][0], 104)

    def test_deepmatch_flag_fallback_is_safe(self):
        model = YouTubeDNNRecall(use_deepmatch=True, epochs=1).fit(self.history, self.embeddings)
        recalled = model.recall(1, topk=3)
        self.assertIsInstance(recalled, list)
        self.assertGreaterEqual(len(recalled), 1)


if __name__ == "__main__":
    unittest.main()
