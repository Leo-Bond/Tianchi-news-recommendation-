"""Recall strategies for the Tianchi news recommendation task.

Available strategies
--------------------
ItemCF  – item-based collaborative filtering using co-click similarity.
BipartiteNetworkRecall – item-item projection recall on user-item bipartite graph.
UserCF  – user-based collaborative filtering.
BPR     – Bayesian Personalised Ranking (matrix factorisation variant).
Word2VecRecall – sequence-based item embedding recall via gensim Word2Vec.
YouTubeDNNRecall       – embedding two-tower proxy recall.
ContentSimilarityRecall – article-content similarity recall.
HotFreshRecall         – popularity + freshness recall.

Each strategy exposes a ``recall(user_id, topk)`` method and returns a list of
``(article_id, score)`` pairs sorted by score (descending).
"""

import math
import os
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


class BipartiteNetworkRecall:
    """Bipartite graph projection recall with IIF/IUF style weighting.

    The similarity formula follows:
      w(i, j) += 1 / (log(|U_i| + 1) * log(|I_u| + 1))
    where |U_i| is the number of users clicked item i and |I_u| is the number
    of items clicked by user u.
    """

    def __init__(
        self,
        topk_sim=20,
        recency_decay=0.7,
        use_iif=True,
        use_last_click_only=True,
    ):
        self.topk_sim = int(topk_sim)
        self.recency_decay = float(recency_decay)
        self.use_iif = bool(use_iif)
        self.use_last_click_only = bool(use_last_click_only)
        self.user_history = {}
        self.item_sim = {}

    @timer
    def fit(self, click_history):
        self.user_history = click_history
        user_item_dict = {u: set(items) for u, items in click_history.items() if items}

        item_user_dict = defaultdict(set)
        for user_id, items in user_item_dict.items():
            for item in items:
                item_user_dict[item].add(user_id)

        sim_item = {}
        for item, users in item_user_dict.items():
            sim_item.setdefault(item, {})
            item_user_count = len(users)
            item_user_penalty = math.log(item_user_count + 1.0)

            for user_id in users:
                user_items = user_item_dict.get(user_id, set())
                user_item_count = len(user_items)
                if user_item_count <= 0:
                    continue

                if self.use_iif:
                    denom = item_user_penalty * math.log(user_item_count + 1.0)
                    increment = 1.0 / denom if denom > 0 else 0.0
                else:
                    increment = 1.0

                for related_item in user_items:
                    if related_item == item:
                        continue
                    sim_item[item].setdefault(related_item, 0.0)
                    sim_item[item][related_item] += increment

        self.item_sim = {}
        for item, related in sim_item.items():
            ranked = sorted(related.items(), key=lambda x: x[1], reverse=True)
            if self.topk_sim > 0:
                ranked = ranked[: self.topk_sim]
            self.item_sim[item] = dict(ranked)

        logger.info("BipartiteNetworkRecall: built similarity for %d items", len(self.item_sim))
        return self

    def recall(self, user_id, topk=50):
        history = self.user_history.get(user_id, [])
        if not history:
            return []

        seen = set(history)
        scores = defaultdict(float)
        if self.use_last_click_only:
            anchor = history[-1]
            for related_item, sim_score in self.item_sim.get(anchor, {}).items():
                if related_item in seen:
                    continue
                scores[related_item] += float(sim_score)
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[: int(topk)]

        n = len(history)
        for idx, item in enumerate(history):
            recency_weight = self.recency_decay ** (n - 1 - idx)
            for related_item, sim_score in self.item_sim.get(item, {}).items():
                if related_item in seen:
                    continue
                scores[related_item] += float(sim_score) * recency_weight

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[: int(topk)]


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


class Word2VecRecall:
    """Word2Vec-based recall trained on user click sequences."""

    def __init__(
        self,
        vector_size=256,
        window=3,
        min_count=1,
        sg=1,
        hs=0,
        seed=2026,
        negative=5,
        workers=10,
        epochs=1,
    ):
        self.vector_size = int(vector_size)
        self.window = int(window)
        self.min_count = int(min_count)
        self.sg = int(sg)
        self.hs = int(hs)
        self.seed = int(seed)
        self.negative = int(negative)
        self.workers = int(workers)
        self.epochs = int(epochs)
        self.user_history = {}
        self._wv = None
        self._backend = "unavailable"

    def fit(self, click_history):
        self.user_history = click_history
        self._wv = None
        self._backend = "unavailable"

        try:
            from gensim.models import Word2Vec
        except Exception as exc:
            logger.warning("Word2VecRecall unavailable: gensim import failed: %s", exc)
            return self

        sentences = []
        for items in click_history.values():
            if not items:
                continue
            sentence = [str(item) for item in items]
            if sentence:
                sentences.append(sentence)

        if not sentences:
            logger.warning("Word2VecRecall skipped: no valid training sequences")
            return self

        try:
            model = Word2Vec(
                sentences=sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg,
                hs=self.hs,
                seed=self.seed,
                negative=self.negative,
                workers=self.workers,
                epochs=self.epochs,
            )
        except TypeError:
            model = Word2Vec(
                sentences=sentences,
                size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg,
                hs=self.hs,
                seed=self.seed,
                negative=self.negative,
                workers=self.workers,
                iter=self.epochs,
            )

        self._wv = model.wv
        self._backend = "gensim"
        logger.info(
            "Word2VecRecall: trained vocab=%d vector_size=%d window=%d epochs=%d",
            len(self._wv),
            self.vector_size,
            self.window,
            self.epochs,
        )
        return self

    def recall(self, user_id, topk=50):
        topk = int(topk)
        if topk <= 0 or self._wv is None:
            return []

        history = self.user_history.get(user_id, [])
        if not history:
            return []

        anchor = str(history[-1])
        if anchor not in self._wv:
            return []

        seen = {str(item) for item in history}
        search_topn = min(len(self._wv), max(topk * 20, len(seen) + topk + 50))
        similar = self._wv.most_similar(anchor, topn=search_topn)

        candidates = []
        for token, score in similar:
            if token in seen:
                continue
            try:
                article_id = int(token)
            except Exception:
                continue
            candidates.append((article_id, float(score)))
            if len(candidates) >= topk:
                break
        return candidates


class YouTubeDNNRecall:
    """PyTorch YouTubeDNN-style recall with a learned user tower and fixed item embeddings."""

    def __init__(
        self,
        recency_decay=0.8,
        embedding_dim=128,
        training_epochs=1,
        batch_size=256,
        faiss_ivf_nlist=4096,
        faiss_ivf_nprobe=32,
        faiss_ivf_min_items=20000,
    ):
        self.recency_decay = recency_decay
        self.embedding_dim = embedding_dim
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.faiss_ivf_nlist = int(faiss_ivf_nlist)
        self.faiss_ivf_nprobe = int(faiss_ivf_nprobe)
        self.faiss_ivf_min_items = int(faiss_ivf_min_items)
        self.user_history = {}
        self.item_embeddings = {}
        self._all_items = []
        self._backend = "unavailable"
        self._item_dim = 0
        self._item_matrix = None
        self._item_ids = []
        self._torch_model = None
        self._torch_device = "cpu"
        self._faiss_index = None
        self._use_faiss = False
        self._faiss_mode = "none"

    def fit(self, click_history, item_embeddings):
        self.user_history = click_history
        self.item_embeddings = item_embeddings or {}
        self._all_items = list(self.item_embeddings.keys())
        self._backend = "unavailable"
        self._faiss_index = None
        self._use_faiss = False
        self._faiss_mode = "none"
        ok = self._fit_torch(click_history)
        if ok:
            self._backend = "pytorch"
            self._build_faiss_index()
        return self

    def _build_faiss_index(self):
        if self._item_matrix is None or self._item_matrix.size == 0:
            return
        try:
            if os.name == "nt":
                os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
                os.environ.setdefault("FAISS_NO_AVX2", "1")
            import faiss

            dim = int(self._item_matrix.shape[1])
            total_items = int(len(self._item_ids))
            item_matrix = self._item_matrix.astype(np.float32)

            use_ivf = total_items >= max(1, self.faiss_ivf_min_items)
            if os.name == "nt" and use_ivf:
                logger.warning(
                    "YouTubeDNNRecall: IVF is disabled on Windows for stability; fallback to FAISS FlatIP."
                )
                use_ivf = False
            if use_ivf:
                nlist = max(1, min(int(self.faiss_ivf_nlist), total_items))
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(item_matrix)
                index.add(item_matrix)
                index.nprobe = max(1, min(int(self.faiss_ivf_nprobe), nlist))
                self._faiss_mode = "ivf"
                self._faiss_index = index
                self._use_faiss = True
                logger.info(
                    "YouTubeDNNRecall: FAISS IVF backend enabled (items=%d, dim=%d, nlist=%d, nprobe=%d)",
                    total_items,
                    dim,
                    nlist,
                    int(index.nprobe),
                )
                return

            index = faiss.IndexFlatIP(dim)
            index.add(item_matrix)
            self._faiss_index = index
            self._use_faiss = True
            self._faiss_mode = "flat"
            logger.info(
                "YouTubeDNNRecall: FAISS FlatIP backend enabled (items=%d, dim=%d)",
                total_items,
                dim,
            )
        except Exception as exc:
            self._faiss_mode = "none"
            logger.info("YouTubeDNNRecall: FAISS unavailable, fallback to NumPy backend: %s", exc)

    def _fit_torch(self, click_history):
        try:
            import torch
            import torch.nn.functional as F
        except Exception as exc:
            logger.warning("YouTubeDNN PyTorch backend unavailable: %s", exc)
            return False

        if not self.item_embeddings:
            logger.warning("YouTubeDNN PyTorch backend unavailable: item embeddings are required.")
            return False

        item_ids = sorted(self.item_embeddings.keys())
        item_matrix = np.stack([safe_normalize(np.asarray(self.item_embeddings[item], dtype=np.float32)) for item in item_ids])
        if item_matrix.ndim != 2:
            logger.warning("YouTubeDNN PyTorch backend unavailable: invalid item embedding shape %s", item_matrix.shape)
            return False

        self._item_ids = item_ids
        self._item_matrix = item_matrix
        self._item_dim = int(item_matrix.shape[1])
        item_lookup = {item_id: idx for idx, item_id in enumerate(self._item_ids)}

        history_vectors = []
        target_vectors = []
        for _, items in click_history.items():
            usable_items = [item for item in items if item in item_lookup]
            for pos in range(1, len(usable_items)):
                hist_vec = self._history_vector(usable_items[:pos])
                if hist_vec is None:
                    continue
                history_vectors.append(hist_vec)
                target_vectors.append(self._item_matrix[item_lookup[usable_items[pos]]])

        if not target_vectors:
            logger.warning("YouTubeDNN PyTorch backend unavailable: no sequential training samples were built.")
            return False

        self._torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.nn.Sequential(
            torch.nn.Linear(self._item_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self._item_dim),
        ).to(self._torch_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        history_tensor = torch.tensor(np.stack(history_vectors), dtype=torch.float32)
        target_tensor = torch.tensor(np.stack(target_vectors), dtype=torch.float32)
        num_samples = int(history_tensor.shape[0])
        indices = np.arange(num_samples)

        for epoch in range(self.training_epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, num_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                hist_batch = history_tensor[batch_idx].to(self._torch_device)
                target_batch = target_tensor[batch_idx].to(self._torch_device)

                user_vec = F.normalize(model(hist_batch), p=2, dim=1)
                item_vec = F.normalize(target_batch, p=2, dim=1)
                logits = user_vec.matmul(item_vec.t()) / 0.07
                labels = torch.arange(logits.size(0), device=self._torch_device)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                batches += 1

            logger.info(
                "YouTubeDNN PyTorch epoch %d/%d loss=%.4f",
                epoch + 1,
                self.training_epochs,
                epoch_loss / max(1, batches),
            )

        self._torch_model = model.eval()
        return True

    def _history_vector(self, history):
        if not history:
            return None
        weighted_vectors = []
        for idx, item in enumerate(history):
            vec = self.item_embeddings.get(item)
            if vec is None:
                continue
            recency_weight = self.recency_decay ** (len(history) - 1 - idx)
            weighted_vectors.append(recency_weight * np.asarray(vec, dtype=np.float32))
        if not weighted_vectors:
            return None
        return safe_normalize(np.mean(weighted_vectors, axis=0)).astype(np.float32)

    def recall(self, user_id, topk=50):
        topk = int(topk)
        if topk <= 0:
            return []
        if self._backend != "pytorch" or self._torch_model is None or self._item_matrix is None:
            return []

        try:
            import torch
            import torch.nn.functional as F
        except Exception:
            return []

        history = self.user_history.get(user_id, [])
        user_hist_vec = self._history_vector(history)
        if user_hist_vec is None:
            return []

        with torch.no_grad():
            user_tensor = torch.tensor(user_hist_vec, dtype=torch.float32, device=self._torch_device).unsqueeze(0)
            user_vec = F.normalize(self._torch_model(user_tensor), p=2, dim=1).cpu().numpy()[0]

        seen = set(history)
        if self._use_faiss and self._faiss_index is not None:
            total_items = len(self._item_ids)
            if total_items <= 0:
                return []
            if self._faiss_mode == "ivf":
                search_k = min(total_items, max(topk * 80, len(history) + topk + 100))
            else:
                search_k = min(total_items, max(topk * 50, len(history) + topk + 100))
            query = np.asarray(user_vec, dtype=np.float32).reshape(1, -1)
            sim_values, indices = self._faiss_index.search(query, search_k)

            candidates = []
            for raw_score, raw_idx in zip(sim_values[0], indices[0]):
                idx = int(raw_idx)
                if idx < 0 or idx >= total_items:
                    continue
                item_id = self._item_ids[idx]
                if item_id in seen:
                    continue
                candidates.append((item_id, float(raw_score)))
            if not candidates:
                return []
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:topk]

        sim = self._item_matrix.dot(user_vec)
        candidates = []
        for item_id, score in zip(self._item_ids, sim):
            if item_id in seen:
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
        self._item_ids = []
        self._item_index = {}
        self._item_matrix = None
        self._item_category_array = None
        self._faiss_index = None
        self._use_faiss = False

    def fit(self, click_history, item_embeddings, item_category):
        self.user_history = click_history
        self.item_embeddings = item_embeddings or {}
        self.item_category = item_category or {}

        self._item_ids = sorted(self.item_embeddings.keys())
        self._item_index = {item_id: idx for idx, item_id in enumerate(self._item_ids)}
        if self._item_ids:
            self._item_matrix = np.stack(
                [safe_normalize(np.asarray(self.item_embeddings[item_id], dtype=np.float32)) for item_id in self._item_ids]
            )
            self._item_category_array = np.asarray(
                [int(self.item_category.get(item_id, -1)) for item_id in self._item_ids],
                dtype=np.int64,
            )
        else:
            self._item_matrix = None
            self._item_category_array = None

        self._faiss_index = None
        self._use_faiss = False
        if self._item_matrix is not None and self._item_matrix.size:
            try:
                import faiss

                dim = int(self._item_matrix.shape[1])
                index = faiss.IndexFlatIP(dim)
                index.add(self._item_matrix.astype(np.float32))
                self._faiss_index = index
                self._use_faiss = True
                logger.info(
                    "ContentSimilarityRecall: FAISS backend enabled (items=%d, dim=%d)",
                    len(self._item_ids),
                    dim,
                )
            except Exception as exc:
                logger.info("ContentSimilarityRecall: FAISS unavailable, fallback to NumPy backend: %s", exc)
        return self

    def recall(self, user_id, topk=50):
        topk = int(topk)
        if topk <= 0:
            return []
        history = self.user_history.get(user_id, [])
        if not history:
            return []
        anchor = history[-1]
        anchor_vec = self.item_embeddings.get(anchor)
        if anchor_vec is None or self._item_matrix is None:
            return []
        anchor_vec = safe_normalize(anchor_vec)
        anchor_cat = self.item_category.get(anchor)
        seen = set(history)

        if self._use_faiss and self._faiss_index is not None:
            total_items = len(self._item_ids)
            if total_items <= 0:
                return []
            search_k = min(total_items, max(topk * 50, len(history) + topk + 100))
            query = np.asarray(anchor_vec, dtype=np.float32).reshape(1, -1)
            sim_values, indices = self._faiss_index.search(query, search_k)

            candidates = []
            for raw_score, raw_idx in zip(sim_values[0], indices[0]):
                idx = int(raw_idx)
                if idx < 0 or idx >= total_items:
                    continue
                item_id = self._item_ids[idx]
                if item_id in seen:
                    continue
                score = float(raw_score)
                if (
                    anchor_cat is not None
                    and self._item_category_array is not None
                    and self._item_category_array[idx] == int(anchor_cat)
                    and self.category_bonus
                ):
                    score += float(self.category_bonus)
                candidates.append((item_id, score))

            if not candidates:
                return []
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:topk]

        scores = self._item_matrix.dot(np.asarray(anchor_vec, dtype=np.float32))
        if anchor_cat is not None and self._item_category_array is not None and self.category_bonus:
            scores = scores + (self._item_category_array == int(anchor_cat)) * float(self.category_bonus)

        seen_indices = [self._item_index[item_id] for item_id in seen if item_id in self._item_index]
        if seen_indices:
            scores = scores.copy()
            scores[np.asarray(seen_indices, dtype=np.int64)] = -np.inf

        available = int(np.isfinite(scores).sum())
        if available <= 0:
            return []

        k = min(int(topk), available)
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(self._item_ids[idx], float(scores[idx])) for idx in top_indices if np.isfinite(scores[idx])]


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
