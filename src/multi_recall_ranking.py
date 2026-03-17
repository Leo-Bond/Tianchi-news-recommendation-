"""Multi-route recall + LightGBM LambdaRank pipeline for Tianchi news recommendation.

Usage
-----
python -m src.multi_recall_ranking --data_dir tcdata --output_dir output
"""

import argparse
import os
import time

import pandas as pd

try:
    from .data_processing import build_user_click_history, split_last_click
    from .ranking import GBDTLRRanker, build_feature_dataframe
    from .recall import (
        BipartiteNetworkRecall,
        ContentSimilarityRecall,
        HotFreshRecall,
        ItemCF,
        Word2VecRecall,
        YouTubeDNNRecall,
        merge_recall_results,
    )
    from .evaluate import make_submission
    from .utils import get_logger
except ImportError:  # support direct script execution
    from data_processing import build_user_click_history, split_last_click
    from ranking import GBDTLRRanker, build_feature_dataframe
    from recall import (
        BipartiteNetworkRecall,
        ContentSimilarityRecall,
        HotFreshRecall,
        ItemCF,
        Word2VecRecall,
        YouTubeDNNRecall,
        merge_recall_results,
    )
    from evaluate import make_submission
    from utils import get_logger

logger = get_logger(__name__, source_file=__file__)


def _log_stage(stage_name, start_time):
    logger.info("%s finished in %.2fs", stage_name, time.time() - start_time)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-route recall + ranking submission generator")
    parser.add_argument("--data_dir", default="tcdata", help="Directory containing click logs")
    parser.add_argument("--output_dir", default="output", help="Directory to save submission")
    parser.add_argument("--topk_recall", type=int, default=10, help="Recall candidates per user")
    parser.add_argument("--topk_submit", type=int, default=5, help="Final submit items per user")
    parser.add_argument("--topk_sim", type=int, default=10, help="ItemCF top similar items kept per item")
    parser.add_argument("--popular_fill_k", type=int, default=100, help="Hot items pool size for fallback")
    parser.add_argument(
        "--recall_weights",
        default="1,0,0,0",
        help="Weights for itemcf,youtube_dnn,content,hot_fresh,w2v,bipartite recalls (4/5/6-value inputs are supported)",
    )
    parser.add_argument(
        "--max_train_users",
        type=int,
        default=0,
        help="Maximum number of training users used to build ranker candidates (0 means all)",
    )
    parser.add_argument("--youtube_dnn_embedding_dim", type=int, default=128, help="Hidden size for the PyTorch YouTubeDNN user tower")
    parser.add_argument("--youtube_dnn_epochs", type=int, default=1, help="Training epochs for the PyTorch YouTubeDNN")
    parser.add_argument("--youtube_dnn_batch_size", type=int, default=256, help="Batch size for the PyTorch YouTubeDNN")
    parser.add_argument("--youtube_dnn_faiss_ivf_nlist", type=int, default=4096, help="FAISS IVF nlist for YouTubeDNN recall")
    parser.add_argument("--youtube_dnn_faiss_ivf_nprobe", type=int, default=32, help="FAISS IVF nprobe for YouTubeDNN recall")
    parser.add_argument("--youtube_dnn_faiss_ivf_min_items", type=int, default=20000, help="Minimum item count to enable IVF instead of FlatIP")
    return parser.parse_args()


def _resolve_test_path(data_dir):
    candidates = [
        "testA_click_log.csv",
        "testB_click_log.csv",
        "test_click_log.csv",
    ]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise IOError(
        "No test click log found under {}. Tried: {}".format(
            data_dir, ", ".join(candidates)
        )
    )


def _build_history(click_df):
    return build_user_click_history(click_df.sort_values(["user_id", "click_timestamp"]))


def _popular_items(click_df, k):
    return click_df["click_article_id"].value_counts().head(k).index.tolist()


def _parse_weights(weight_text):
    weights = [float(x) for x in weight_text.split(",") if x.strip()]
    if len(weights) == 4:
        weights.extend([0.0, 0.0])
    elif len(weights) == 5:
        weights.append(0.0)
    if len(weights) != 6:
        logger.warning("Invalid --recall_weights '%s', fallback to default 0.4,0.2,0.2,0.2,0.0,0.0", weight_text)
        return [0.4, 0.2, 0.2, 0.2, 0.0, 0.0]
    total = sum(weights)
    if total <= 0:
        logger.warning("Non-positive --recall_weights sum for '%s', fallback to default", weight_text)
        return [0.4, 0.2, 0.2, 0.2, 0.0, 0.0]
    return [w / total for w in weights]


def _active_routes(weights):
    route_names = ["itemcf", "youtube_dnn", "content", "hot_fresh", "w2v", "bipartite"]
    return [(name, weight) for name, weight in zip(route_names, weights) if weight > 0]


def _load_optional_frame(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _build_item_embeddings(emb_df):
    if emb_df.empty or "article_id" not in emb_df.columns:
        return {}
    emb_cols = [c for c in emb_df.columns if c != "article_id"]
    if not emb_cols:
        return {}
    return {
        int(row["article_id"]): row[emb_cols].to_numpy(dtype=float)
        for _, row in emb_df.iterrows()
    }


def _build_item_meta(click_df, articles_df):
    pop_count = click_df["click_article_id"].value_counts()
    if pop_count.empty:
        item_popularity = {}
    else:
        max_pop = float(pop_count.max())
        item_popularity = {int(k): float(v) / max_pop for k, v in pop_count.items()}

    item_category, item_created_at = {}, {}
    if not articles_df.empty and "article_id" in articles_df.columns:
        if "category_id" in articles_df.columns:
            item_category = {
                int(row["article_id"]): int(row["category_id"])
                for _, row in articles_df[["article_id", "category_id"]].dropna().iterrows()
            }
        if "created_at_ts" in articles_df.columns:
            item_created_at = {
                int(row["article_id"]): float(row["created_at_ts"])
                for _, row in articles_df[["article_id", "created_at_ts"]].dropna().iterrows()
            }
    return item_popularity, item_category, item_created_at


def _build_user_last_timestamp(click_df):
    if click_df.empty:
        return {}
    last = click_df.sort_values(["user_id", "click_timestamp"]).groupby("user_id").tail(1)
    return {int(row["user_id"]): float(row["click_timestamp"]) for _, row in last.iterrows()}


def _fit_recall_models(
    history,
    topk_sim,
    item_embeddings,
    item_category,
    item_popularity,
    item_created_at,
    youtube_dnn_embedding_dim=128,
    youtube_dnn_epochs=1,
    youtube_dnn_batch_size=256,
    youtube_dnn_faiss_ivf_nlist=4096,
    youtube_dnn_faiss_ivf_nprobe=32,
    youtube_dnn_faiss_ivf_min_items=20000,
    active_routes=None,
):
    active_routes = active_routes or [("itemcf", 1.0)]
    recallers = []
    for route_name, _ in active_routes:
        if route_name == "itemcf":
            recallers.append(ItemCF(topk_sim=topk_sim).fit(history))
        elif route_name == "bipartite":
            recallers.append(BipartiteNetworkRecall(topk_sim=topk_sim, use_iif=True).fit(history))
        elif route_name == "w2v":
            recallers.append(Word2VecRecall().fit(history))
        elif route_name == "youtube_dnn":
            recallers.append(
                YouTubeDNNRecall(
                    embedding_dim=youtube_dnn_embedding_dim,
                    training_epochs=youtube_dnn_epochs,
                    batch_size=youtube_dnn_batch_size,
                    faiss_ivf_nlist=youtube_dnn_faiss_ivf_nlist,
                    faiss_ivf_nprobe=youtube_dnn_faiss_ivf_nprobe,
                    faiss_ivf_min_items=youtube_dnn_faiss_ivf_min_items,
                ).fit(history, item_embeddings=item_embeddings)
            )
        elif route_name == "content":
            recallers.append(
                ContentSimilarityRecall().fit(
                    history, item_embeddings=item_embeddings, item_category=item_category
                )
            )
        elif route_name == "hot_fresh":
            recallers.append(
                HotFreshRecall().fit(
                    history,
                    item_popularity=item_popularity,
                    item_created_at=item_created_at,
                )
            )
    return recallers


def _multi_recall(
    users,
    recallers,
    history,
    topk_recall,
    route_weights,
    popular_items,
):
    results = {}
    total_users = len(users)
    for idx, user_id in enumerate(users, start=1):
        route_results = [{user_id: rec.recall(user_id, topk=topk_recall)} for rec in recallers]
        merged = merge_recall_results(route_results, weights=route_weights).get(user_id, [])
        merged = merged[:topk_recall]

        if len(merged) < topk_recall:
            seen = set(history.get(user_id, []))
            existing = {item for item, _ in merged}
            fill_score = -1.0
            for item in popular_items:
                if item in seen or item in existing:
                    continue
                merged.append((item, fill_score))
                fill_score -= 1.0
                if len(merged) >= topk_recall:
                    break
        results[user_id] = merged
        if idx % 5000 == 0 or idx == total_users:
            logger.info("Multi recall progress: %d/%d users", idx, total_users)
    return results


def build_baseline_submission(
    data_dir,
    output_dir,
    topk_recall,
    topk_submit,
    topk_sim,
    popular_fill_k,
    recall_weights="0.4,0.2,0.2,0.2",
    max_train_users=0,
    youtube_dnn_embedding_dim=128,
    youtube_dnn_epochs=1,
    youtube_dnn_batch_size=256,
    youtube_dnn_faiss_ivf_nlist=4096,
    youtube_dnn_faiss_ivf_nprobe=32,
    youtube_dnn_faiss_ivf_min_items=20000,
):
    total_start = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stage_start = time.time()
    train_path = os.path.join(data_dir, "train_click_log.csv")
    test_path = _resolve_test_path(data_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    articles_df = _load_optional_frame(os.path.join(data_dir, "articles.csv"))
    emb_df = _load_optional_frame(os.path.join(data_dir, "articles_emb.csv"))
    logger.info("Loaded train: %s shape=%s", train_path, train_df.shape)
    logger.info("Loaded test: %s shape=%s", test_path, test_df.shape)
    _log_stage("Data loading", stage_start)

    stage_start = time.time()
    all_click = pd.concat([train_df, test_df], ignore_index=True)
    all_click.sort_values(["user_id", "click_timestamp"], inplace=True)
    item_embeddings = _build_item_embeddings(emb_df)
    item_popularity, item_category, item_created_at = _build_item_meta(all_click, articles_df)
    weights = _parse_weights(recall_weights)
    active_routes = _active_routes(weights)
    active_weights = [weight for _, weight in active_routes]
    hot_items = _popular_items(all_click, popular_fill_k)
    logger.info(
        "Active recall routes: %s",
        ", ".join("{}={:.3f}".format(name, weight) for name, weight in active_routes),
    )
    _log_stage("Global metadata build", stage_start)

    # ranker training data: leave-one-out on train clicks
    stage_start = time.time()
    hist_df, label_df = split_last_click(train_df)
    train_history = _build_history(hist_df)
    train_users = label_df["user_id"].drop_duplicates().tolist()
    if max_train_users and len(train_users) > max_train_users:
        train_users = train_users[:max_train_users]
        logger.info("Train users truncated to %d for lightweight ranking", len(train_users))
    train_last_ts = _build_user_last_timestamp(hist_df)
    _log_stage("Train split and history build", stage_start)

    stage_start = time.time()
    train_recallers = _fit_recall_models(
        history=train_history,
        topk_sim=topk_sim,
        item_embeddings=item_embeddings,
        item_category=item_category,
        item_popularity=item_popularity,
        item_created_at=item_created_at,
        active_routes=active_routes,
        youtube_dnn_embedding_dim=youtube_dnn_embedding_dim,
        youtube_dnn_epochs=youtube_dnn_epochs,
        youtube_dnn_batch_size=youtube_dnn_batch_size,
        youtube_dnn_faiss_ivf_nlist=youtube_dnn_faiss_ivf_nlist,
        youtube_dnn_faiss_ivf_nprobe=youtube_dnn_faiss_ivf_nprobe,
        youtube_dnn_faiss_ivf_min_items=youtube_dnn_faiss_ivf_min_items,
    )
    _log_stage("Train recall model fit", stage_start)

    stage_start = time.time()
    train_candidates = _multi_recall(
        users=train_users,
        recallers=train_recallers,
        history=train_history,
        topk_recall=topk_recall,
        route_weights=active_weights,
        popular_items=hot_items,
    )
    _log_stage("Train candidate recall", stage_start)

    stage_start = time.time()
    train_feature_df = build_feature_dataframe(
        candidates_by_user=train_candidates,
        user_history=train_history,
        item_embeddings=item_embeddings,
        item_category=item_category,
        item_created_at=item_created_at,
        item_popularity=item_popularity,
        user_last_ts=train_last_ts,
    )
    _log_stage("Train feature build", stage_start)
    if train_feature_df.empty:
        logger.warning("No train features built; fallback to recall score ranking.")

    train_labels = label_df.rename(columns={"click_article_id": "article_id"})[
        ["user_id", "article_id"]
    ].copy()
    train_labels["label"] = 1

    stage_start = time.time()
    ranker = GBDTLRRanker().fit(train_feature_df, train_labels) if not train_feature_df.empty else None
    _log_stage("Ranker fit", stage_start)

    # inference recall on all-click history
    stage_start = time.time()
    user_history = _build_history(all_click)
    test_users = test_df["user_id"].drop_duplicates().tolist()
    test_last_ts = _build_user_last_timestamp(all_click)
    _log_stage("Test history build", stage_start)

    stage_start = time.time()
    recallers = _fit_recall_models(
        history=user_history,
        topk_sim=topk_sim,
        item_embeddings=item_embeddings,
        item_category=item_category,
        item_popularity=item_popularity,
        item_created_at=item_created_at,
        active_routes=active_routes,
        youtube_dnn_embedding_dim=youtube_dnn_embedding_dim,
        youtube_dnn_epochs=youtube_dnn_epochs,
        youtube_dnn_batch_size=youtube_dnn_batch_size,
        youtube_dnn_faiss_ivf_nlist=youtube_dnn_faiss_ivf_nlist,
        youtube_dnn_faiss_ivf_nprobe=youtube_dnn_faiss_ivf_nprobe,
        youtube_dnn_faiss_ivf_min_items=youtube_dnn_faiss_ivf_min_items,
    )
    _log_stage("Test recall model fit", stage_start)

    stage_start = time.time()
    test_candidates = _multi_recall(
        users=test_users,
        recallers=recallers,
        history=user_history,
        topk_recall=topk_recall,
        route_weights=active_weights,
        popular_items=hot_items,
    )
    _log_stage("Test candidate recall", stage_start)

    stage_start = time.time()
    test_feature_df = build_feature_dataframe(
        candidates_by_user=test_candidates,
        user_history=user_history,
        item_embeddings=item_embeddings,
        item_category=item_category,
        item_created_at=item_created_at,
        item_popularity=item_popularity,
        user_last_ts=test_last_ts,
    )
    _log_stage("Test feature build", stage_start)

    stage_start = time.time()
    if ranker is not None and not test_feature_df.empty:
        ranked_df = ranker.predict(test_feature_df)
    else:
        rows = []
        for user_id, cands in test_candidates.items():
            for article_id, score in cands:
                rows.append((user_id, article_id, score))
        ranked_df = pd.DataFrame(rows, columns=["user_id", "article_id", "rank_score"])
    _log_stage("Ranking", stage_start)

    stage_start = time.time()
    submission = make_submission(ranked_df, topk=topk_submit)

    submit_path = os.path.join(output_dir, "submission_multi_recall_ranking.csv")
    submission.to_csv(submit_path, index=False)
    logger.info("Saved baseline submission to %s shape=%s", submit_path, submission.shape)
    _log_stage("Submission export", stage_start)
    _log_stage("Full pipeline", total_start)
    return submit_path


def main():
    args = parse_args()
    build_baseline_submission(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        topk_recall=args.topk_recall,
        topk_submit=args.topk_submit,
        topk_sim=args.topk_sim,
        popular_fill_k=args.popular_fill_k,
        recall_weights=args.recall_weights,
        max_train_users=args.max_train_users,
        youtube_dnn_embedding_dim=args.youtube_dnn_embedding_dim,
        youtube_dnn_epochs=args.youtube_dnn_epochs,
        youtube_dnn_batch_size=args.youtube_dnn_batch_size,
        youtube_dnn_faiss_ivf_nlist=args.youtube_dnn_faiss_ivf_nlist,
        youtube_dnn_faiss_ivf_nprobe=args.youtube_dnn_faiss_ivf_nprobe,
        youtube_dnn_faiss_ivf_min_items=args.youtube_dnn_faiss_ivf_min_items,
    )


if __name__ == "__main__":
    main()
