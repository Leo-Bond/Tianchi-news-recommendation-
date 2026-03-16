import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        model = YouTubeDNNRecall(training_epochs=1, batch_size=2).fit(self.history, self.embeddings)
        recalled = model.recall(1, topk=3)
        self.assertEqual(model._backend, "pytorch")
        self.assertGreaterEqual(len(recalled), 1)
        self.assertIn(recalled[0][0], {103, 104})

    def test_pytorch_backend_is_safe(self):
        model = YouTubeDNNRecall(training_epochs=1, batch_size=2).fit(self.history, self.embeddings)
        recalled = model.recall(1, topk=3)
        self.assertEqual(model._backend, "pytorch")
        self.assertIsInstance(recalled, list)
        self.assertGreaterEqual(len(recalled), 1)


if __name__ == "__main__":
    unittest.main()
