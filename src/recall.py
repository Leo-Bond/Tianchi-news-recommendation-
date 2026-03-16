"""Recall strategies for the Tianchi news recommendation task.

Available strategies
--------------------
ItemCF  – item-based collaborative filtering using co-click similarity.
UserCF  – user-based collaborative filtering.
BPR     – Bayesian Personalised Ranking (matrix factorisation variant).
YouTubeDNNRecall       – embedding two-tower proxy recall.
ContentSimilarityRecall – article-content similarity recall.
HotFreshRecall         – popularity + freshness recall.

Each strategy exposes a ``recall(user_id, topk)`` method and returns a list of
``(article_id, score)`` pairs sorted by score (descending).
"""

import math
from collections import defaultdict

import numpy as np

from .utils import get_logger, safe_normalize, timer

logger = get_logger(__name__)

RecallResult = list


# ---------------------------------------------------------------------------
# ItemCF
# ---------------------------------------------------------------------------

class ItemCF:
    """Item-based collaborative filtering.

    Similarity is computed via co-click counts normalised by item popularity
    (IUF weighting – inverse user frequency), similar to the formula used in
    many Tianchi baselines.
    """

    def __init__(self, topk_sim=10):
        """
        Parameters
        ----------
        topk_sim : int
            Number of most-similar items to keep per item.
        """
        self.topk_sim = topk_sim
        self.item_sim = {}
        self.user_history = {}

    @timer
    def fit(self, click_history):
        """Compute item–item similarity from click histories.

        Parameters
        ----------
        click_history : dict
            {user_id -> [article_id, ...]} ordered by click time.
        """
        self.user_history = click_history

        # co-click count matrix (sparse, stored as nested dict)
        item_user_count = defaultdict(int)
        co_count = defaultdict(lambda: defaultdict(float))

        for user_id, items in click_history.items():
            n = len(items)
            iuf_weight = 1.0 / math.log(1 + n)  # inverse user frequency
            for i, item_i in enumerate(items):
                item_user_count[item_i] += 1
                for j, item_j in enumerate(items):
                    if item_i == item_j:
                        continue
                    # positional decay: prefer items clicked close together
                    pos_weight = 1.0 / (1 + abs(i - j))
                    co_count[item_i][item_j] += iuf_weight * pos_weight

        # normalise by sqrt of individual item popularities
        self.item_sim = {}
        for item_i, related in co_count.items():
            self.item_sim[item_i] = {}
            for item_j, cnt in related.items():
                denom = math.sqrt(item_user_count[item_i] * item_user_count[item_j])
                self.item_sim[item_i][item_j] = cnt / denom if denom > 0 else 0.0

            # keep only topk most similar items
            self.item_sim[item_i] = dict(
                sorted(self.item_sim[item_i].items(), key=lambda x: x[1], reverse=True)[
                    : self.topk_sim
                ]
            )

        logger.info("ItemCF: built similarity for %d items", len(self.item_sim))
        return self

    def recall(self, user_id, topk=50):
        """Return top-*k* article candidates for *user_id*."""
        history = self.user_history.get(user_id, [])
        if not history:
            return []

        scores = defaultdict(float)
        history_set = set(history)
        n = len(history)

        for idx, item in enumerate(history):
            # more recent clicks get higher weight
            recency_weight = 0.7 ** (n - 1 - idx)
            for sim_item, sim_score in self.item_sim.get(item, {}).items():
                if sim_item in history_set:
                    continue
                scores[sim_item] += sim_score * recency_weight

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]


# ---------------------------------------------------------------------------
# UserCF
# ---------------------------------------------------------------------------

class UserCF:
    """User-based collaborative filtering.

    Finds the most similar users (based on co-clicked articles) and
    recommends articles they clicked that the target user has not seen.
    """

    def __init__(self, topk_users=20):
        self.topk_users = topk_users
        self.user_sim = {}
        self.user_history = {}

    @timer
    def fit(self, click_history):
        """Compute user–user similarity."""
        self.user_history = click_history

        item_users = defaultdict(list)
        for user_id, items in click_history.items():
            for item in items:
                item_users[item].append(user_id)

        user_click_count = {u: len(items) for u, items in click_history.items()}
        co_count = defaultdict(lambda: defaultdict(float))

        for item, users in item_users.items():
            iif_weight = 1.0 / math.log(1 + len(users))  # inverse item frequency
            for u_i in users:
                for u_j in users:
                    if u_i == u_j:
                        continue
                    co_count[u_i][u_j] += iif_weight

        self.user_sim = {}
        for u_i, related in co_count.items():
            self.user_sim[u_i] = {}
            for u_j, cnt in related.items():
                denom = math.sqrt(user_click_count[u_i] * user_click_count[u_j])
                self.user_sim[u_i][u_j] = cnt / denom if denom > 0 else 0.0
            self.user_sim[u_i] = dict(
                sorted(self.user_sim[u_i].items(), key=lambda x: x[1], reverse=True)[
                    : self.topk_users
                ]
            )

        logger.info("UserCF: built similarity for %d users", len(self.user_sim))
        return self

    def recall(self, user_id, topk=50):
        """Return top-*k* article candidates for *user_id*."""
        history_set = set(self.user_history.get(user_id, []))
        scores = defaultdict(float)

        for sim_user, sim_score in self.user_sim.get(user_id, {}).items():
            for item in self.user_history.get(sim_user, []):
                if item in history_set:
                    continue
                scores[item] += sim_score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]


# ---------------------------------------------------------------------------
# BPR (Bayesian Personalised Ranking)
# ---------------------------------------------------------------------------

class BPR:
    """Bayesian Personalised Ranking (stochastic gradient descent version).

    Learns user and item latent vectors by optimising the BPR criterion:
    maximise the probability that a clicked item is ranked above an
    un-clicked item for each user.
    """

    def __init__(
        self,
        n_factors=32,
        n_epochs=20,
        lr=0.01,
        reg=0.001,
        seed=42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

        self.user_factors = None
        self.item_factors = None
        self._user_index = {}
        self._item_index = {}
        self.user_history = {}

    @timer
    def fit(self, click_history):
        """Train BPR on *click_history*."""
        self.user_history = click_history
        rng = np.random.RandomState(self.seed)

        users = list(click_history.keys())
        items = sorted({item for items in click_history.values() for item in items})

        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {it: i for i, it in enumerate(items)}
        n_users, n_items = len(users), len(items)

        self.user_factors = rng.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.n_factors))

        # build positive item lists per user (indexed)
        pos_items = {
            self._user_index[u]: [self._item_index[it] for it in itms]
            for u, itms in click_history.items()
        }

        for epoch in range(self.n_epochs):
            loss = 0.0
            for u_idx, pos_list in pos_items.items():
                if not pos_list:
                    continue
                # sample a positive item
                pos_idx = pos_list[rng.randint(len(pos_list))]
                # sample a negative item (not in positive set)
                neg_idx = rng.randint(n_items)
                while neg_idx in pos_list:
                    neg_idx = rng.randint(n_items)

                u_vec = self.user_factors[u_idx]
                p_vec = self.item_factors[pos_idx]
                n_vec = self.item_factors[neg_idx]

                diff = np.dot(u_vec, p_vec - n_vec)
                sigmoid = 1.0 / (1.0 + math.exp(-diff))
                grad = 1.0 - sigmoid

                self.user_factors[u_idx] += self.lr * (grad * (p_vec - n_vec) - self.reg * u_vec)
                self.item_factors[pos_idx] += self.lr * (grad * u_vec - self.reg * p_vec)
                self.item_factors[neg_idx] += self.lr * (-grad * u_vec - self.reg * n_vec)

                loss -= math.log(sigmoid + 1e-9)

            logger.info("BPR epoch %d/%d  loss=%.4f", epoch + 1, self.n_epochs, loss)

        return self

    def recall(self, user_id, topk=50):
        """Return top-*k* article candidates for *user_id*."""
        if user_id not in self._user_index:
            return []

        u_idx = self._user_index[user_id]
        scores = self.item_factors.dot(self.user_factors[u_idx])
        history_set = {
            self._item_index[it]
            for it in self.user_history.get(user_id, [])
            if it in self._item_index
        }

        # zero out already-seen items
        for idx in history_set:
            scores[idx] = -np.inf

        n_candidates = len(scores) - len(history_set)
        if n_candidates <= 0:
            return []
        k = min(topk, n_candidates)
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        index_to_item = {v: k for k, v in self._item_index.items()}
        return [
            (index_to_item[i], float(scores[i]))
            for i in top_indices
            if i in index_to_item and scores[i] != -np.inf
        ]


class YouTubeDNNRecall:
    """YouTubeDNN recall with DeepMatch/DeepCTR/TensorFlow backend and safe fallback."""

    def __init__(
        self,
        recency_decay=0.8,
        use_deepmatch=True,
        embedding_dim=16,
        training_epochs=1,
        batch_size=256,
    ):
        """Initialize the recall model.

        Parameters
        ----------
        recency_decay : float
            Exponential decay factor (0~1] for user-history weighting.
            Higher values decay more slowly, keeping older clicks more influential.
        use_deepmatch : bool
            Whether to try DeepMatch + DeepCTR + TensorFlow backend first.
        embedding_dim : int
            Embedding size for the DeepMatch YouTubeDNN backend.
        training_epochs : int
            Training epochs for the DeepMatch YouTubeDNN backend.
        batch_size : int
            Batch size for the DeepMatch YouTubeDNN backend.
        """
        if recency_decay <= 0 or recency_decay > 1:
            raise ValueError("recency_decay must be greater than 0 and less than or equal to 1.")
        self.recency_decay = recency_decay
        self.use_deepmatch = use_deepmatch
        self.embedding_dim = embedding_dim
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.user_history = {}
        self.item_embeddings = {}
        self._all_items = []
        self._backend = "proxy"

        # deepmatch backend artifacts
        self._user2idx = {}
        self._item2idx = {}
        self._idx2item = {}
        self._seq_max_len = 1
        self._dm_user_model = None
        self._dm_item_matrix = None
        self._dm_item_ids = []

    def fit(self, click_history, item_embeddings):
        self.user_history = click_history
        self.item_embeddings = item_embeddings or {}
        self._all_items = list(self.item_embeddings.keys())
        self._backend = "proxy"
        if self.use_deepmatch:
            ok = self._fit_deepmatch(click_history)
            if ok:
                self._backend = "deepmatch"
        return self

    def _fit_deepmatch(self, click_history):
        try:
            from deepctr.feature_column import SparseFeat, VarLenSparseFeat
            from deepmatch.models import YoutubeDNN
            from deepmatch.utils import sampledsoftmaxloss
            from tensorflow.keras.models import Model
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except Exception as exc:
            logger.warning("YouTubeDNN deepmatch backend unavailable, fallback to proxy: %s", exc)
            return False

        users = [u for u, items in click_history.items() if len(items) >= 2]
        if not users:
            return False

        all_items = sorted({it for items in click_history.values() for it in items})
        if not all_items:
            return False

        self._user2idx = {u: idx + 1 for idx, u in enumerate(users)}
        self._item2idx = {it: idx + 1 for idx, it in enumerate(all_items)}
        self._idx2item = {idx: item for item, idx in self._item2idx.items()}

        seqs, lens, user_ids, targets = [], [], [], []
        for user_id in users:
            mapped = [self._item2idx[it] for it in click_history.get(user_id, []) if it in self._item2idx]
            for pos in range(1, len(mapped)):
                hist = mapped[:pos]
                seqs.append(hist)
                lens.append(len(hist))
                user_ids.append(self._user2idx[user_id])
                targets.append(mapped[pos])

        if not targets:
            return False

        self._seq_max_len = max(lens) if lens else 1
        hist_padded = pad_sequences(
            seqs, maxlen=self._seq_max_len, padding="post", truncating="post", value=0
        )

        user_feature_columns = [
            SparseFeat("user_id", vocabulary_size=len(self._user2idx) + 1, embedding_dim=self.embedding_dim),
            VarLenSparseFeat(
                SparseFeat(
                    "hist_article_id",
                    vocabulary_size=len(self._item2idx) + 1,
                    embedding_dim=self.embedding_dim,
                    embedding_name="article_id",
                ),
                maxlen=self._seq_max_len,
                combiner="mean",
                length_name="hist_len",
            ),
        ]
        item_feature_columns = [
            SparseFeat("article_id", vocabulary_size=len(self._item2idx) + 1, embedding_dim=self.embedding_dim)
        ]
        model = YoutubeDNN(
            user_feature_columns,
            item_feature_columns,
            num_sampled=min(5, len(self._item2idx)),
            user_dnn_hidden_units=(64, self.embedding_dim),
        )
        model.compile("adam", sampledsoftmaxloss)

        train_input = {
            "user_id": np.array(user_ids, dtype="int32"),
            "hist_article_id": np.array(hist_padded, dtype="int32"),
            "hist_len": np.array(lens, dtype="int32"),
            "article_id": np.array(targets, dtype="int32"),
        }
        y = np.array(targets, dtype="int32")
        model.fit(
            train_input,
            y,
            batch_size=self.batch_size,
            epochs=self.training_epochs,
            verbose=0,
        )

        self._dm_user_model = Model(inputs=model.user_input, outputs=model.user_embedding)
        dm_item_model = Model(inputs=model.item_input, outputs=model.item_embedding)
        item_idx = np.array(sorted(self._idx2item.keys()), dtype="int32")
        self._dm_item_ids = item_idx.tolist()
        item_vec = dm_item_model.predict({"article_id": item_idx}, verbose=0)
        norms = np.linalg.norm(item_vec, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self._dm_item_matrix = item_vec / norms
        return True

    def _user_embedding(self, user_id):
        history = self.user_history.get(user_id, [])
        if not history:
            return None
        vectors = []
        for idx, item in enumerate(history):
            vec = self.item_embeddings.get(item)
            if vec is None:
                continue
            recency_weight = self.recency_decay ** (len(history) - 1 - idx)
            vectors.append(recency_weight * vec)
        if not vectors:
            return None
        return safe_normalize(np.mean(vectors, axis=0))

    def recall(self, user_id, topk=50):
        if self._backend == "deepmatch":
            return self._recall_deepmatch(user_id, topk=topk)

        user_vec = self._user_embedding(user_id)
        if user_vec is None:
            return []
        seen = set(self.user_history.get(user_id, []))
        scores = []
        for item in self._all_items:
            if item in seen:
                continue
            item_vec = self.item_embeddings[item]
            score = float(np.dot(user_vec, safe_normalize(item_vec)))
            scores.append((item, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

    def _recall_deepmatch(self, user_id, topk=50):
        if user_id not in self._user2idx or self._dm_user_model is None or self._dm_item_matrix is None:
            return []
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except Exception:
            return []

        history = self.user_history.get(user_id, [])
        mapped = [self._item2idx[it] for it in history if it in self._item2idx]
        if not mapped:
            return []
        hist_padded = pad_sequences(
            [mapped], maxlen=self._seq_max_len, padding="post", truncating="post", value=0
        )
        user_vec = self._dm_user_model.predict(
            {
                "user_id": np.array([self._user2idx[user_id]], dtype="int32"),
                "hist_article_id": np.array(hist_padded, dtype="int32"),
                "hist_len": np.array([len(mapped)], dtype="int32"),
            },
            verbose=0,
        )[0]
        user_vec = safe_normalize(user_vec)
        sim = self._dm_item_matrix.dot(user_vec)
        seen = set(history)
        candidates = []
        for item_idx, score in zip(self._dm_item_ids, sim):
            item_id = self._idx2item.get(item_idx)
            if item_id is None or item_id in seen:
                continue
            candidates.append((item_id, float(score)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:topk]


class ContentSimilarityRecall:
    """Content-similarity recall based on item embeddings and category match."""

    def __init__(self, category_bonus=0.1):
        """Initialize the recall model.

        Parameters
        ----------
        category_bonus : float
            Additive bonus for same-category candidates.
            Typical values are in [0, 1].
        """
        self.category_bonus = category_bonus
        self.user_history = {}
        self.item_embeddings = {}
        self.item_category = {}

    def fit(self, click_history, item_embeddings, item_category):
        self.user_history = click_history
        self.item_embeddings = item_embeddings or {}
        self.item_category = item_category or {}
        return self

    def recall(self, user_id, topk=50):
        history = self.user_history.get(user_id, [])
        if not history:
            return []
        anchor = history[-1]
        anchor_vec = self.item_embeddings.get(anchor)
        if anchor_vec is None:
            return []
        anchor_vec = safe_normalize(anchor_vec)
        anchor_cat = self.item_category.get(anchor)
        seen = set(history)
        scores = []
        for item, emb in self.item_embeddings.items():
            if item in seen:
                continue
            cos = float(np.dot(anchor_vec, safe_normalize(emb)))
            same_category = anchor_cat is not None and self.item_category.get(item) == anchor_cat
            cat_bonus = self.category_bonus if same_category else 0.0
            scores.append((item, cos + cat_bonus))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]


class HotFreshRecall:
    """Popularity + freshness recall."""

    def __init__(self, fresh_weight=0.3):
        """Initialize the recall model.

        Parameters
        ----------
        fresh_weight : float
            Weight assigned to freshness score in [0, 1].
            Popularity weight becomes (1 - fresh_weight).
        """
        if fresh_weight < 0 or fresh_weight > 1:
            raise ValueError("fresh_weight must be between 0 and 1 (inclusive).")
        self.fresh_weight = fresh_weight
        self.user_history = {}
        self.item_pop = {}
        self.item_fresh = {}
        self._all_items = []

    def fit(self, click_history, item_popularity, item_created_at):
        self.user_history = click_history
        self.item_pop = item_popularity or {}
        item_created_at = item_created_at or {}
        created_values = list(item_created_at.values()) if item_created_at else []
        self._all_items = sorted(set(self.item_pop.keys()) | set(item_created_at.keys()))
        if created_values:
            mn, mx = min(created_values), max(created_values)
            denom = (mx - mn) if mx != mn else 1.0
            self.item_fresh = {
                item: (item_created_at.get(item, mn) - mn) / denom for item in self._all_items
            }
        else:
            self.item_fresh = {item: 0.0 for item in self._all_items}
        return self

    def recall(self, user_id, topk=50):
        seen = set(self.user_history.get(user_id, []))
        scores = []
        for item in self._all_items:
            if item in seen:
                continue
            pop = self.item_pop.get(item, 0.0)
            fresh = self.item_fresh.get(item, 0.0)
            score = (1.0 - self.fresh_weight) * pop + self.fresh_weight * fresh
            scores.append((item, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]


# ---------------------------------------------------------------------------
# Ensemble helper
# ---------------------------------------------------------------------------

def merge_recall_results(results_list, weights=None):
    """Merge recall results from multiple strategies.

    Scores are linearly combined (weighted sum) after min-max normalisation
    per user per strategy.

    Parameters
    ----------
    results_list : list of dicts
        Each dict maps user_id -> [(article_id, score), ...].
    weights : list of float, optional
        Per-strategy weights; defaults to uniform.

    Returns
    -------
    dict : {user_id -> [(article_id, merged_score), ...]} sorted by score.
    """
    n = len(results_list)
    if weights is None:
        weights = [1.0 / n] * n

    all_users = set()
    for r in results_list:
        all_users |= set(r.keys())

    merged = defaultdict(lambda: defaultdict(float))

    for weight, result in zip(weights, results_list):
        for user_id, candidates in result.items():
            if not candidates:
                continue
            scores = [s for _, s in candidates]
            min_s, max_s = min(scores), max(scores)
            denom = (max_s - min_s) if max_s != min_s else 1.0
            for article_id, score in candidates:
                norm = (score - min_s) / denom
                merged[user_id][article_id] += weight * norm

    return {
        user_id: sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for user_id, scores in merged.items()
    }
