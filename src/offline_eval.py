"""Offline evaluation on train set by leave-one-out split.

Usage
-----
python -m src.offline_eval --data_dir tcdata
"""

import argparse
import os

import pandas as pd

try:
    from .data_processing import load_click_log, split_last_click, build_user_click_history
    from .ranking import GBDTLRRanker, build_feature_dataframe
    from .recall import (
        ItemCF,
        YouTubeDNNRecall,
        ContentSimilarityRecall,
        HotFreshRecall,
        merge_recall_results,
    )
    from .evaluate import evaluate
    from .utils import get_logger
except ImportError:
    from data_processing import load_click_log, split_last_click, build_user_click_history
    from ranking import GBDTLRRanker, build_feature_dataframe
    from recall import (
        ItemCF,
        YouTubeDNNRecall,
        ContentSimilarityRecall,
        HotFreshRecall,
        merge_recall_results,
    )
    from evaluate import evaluate
    from utils import get_logger

logger = get_logger(__name__, source_file=__file__)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline evaluation on train_click_log.csv")
    parser.add_argument("--data_dir", default="tcdata", help="Directory containing train_click_log.csv")
    parser.add_argument("--topk_recall", type=int, default=50, help="Recall candidates per user")
    parser.add_argument("--topk_sim", type=int, default=20, help="ItemCF top similar items kept per item")
    parser.add_argument("--popular_fill_k", type=int, default=200, help="Hot items pool size for fallback")
    parser.add_argument("--k", type=int, default=5, help="Metric cutoff k")
    parser.add_argument(
        "--recall_weights",
        default="1,0,0,0",
        help="Weights for itemcf,youtube_dnn,content,hot_fresh recalls",
    )
    parser.add_argument("--youtube_dnn_embedding_dim", type=int, default=128, help="Hidden size for YouTubeDNN")
    parser.add_argument("--youtube_dnn_epochs", type=int, default=1, help="Training epochs for YouTubeDNN")
    parser.add_argument("--youtube_dnn_batch_size", type=int, default=256, help="Batch size for YouTubeDNN")
    parser.add_argument(
        "--with_ranking",
        action="store_true",
        help="Enable GBDT+LR ranking after multi-route recall",
    )
    parser.add_argument(
        "--max_train_users",
        type=int,
        default=20000,
        help="Maximum users used to train offline ranker (0 means all)",
    )
    return parser.parse_args()


def _popular_items(click_df, k):
    return click_df["click_article_id"].value_counts().head(k).index.tolist()


def _parse_weights(weight_text):
    weights = [float(x) for x in weight_text.split(",") if x.strip()]
    if len(weights) != 4:
        logger.warning("Invalid --recall_weights '%s', fallback to 1,0,0,0", weight_text)
        return [1.0, 0.0, 0.0, 0.0]
    total = sum(weights)
    if total <= 0:
        logger.warning("Non-positive --recall_weights sum, fallback to 1,0,0,0")
        return [1.0, 0.0, 0.0, 0.0]
    return [w / total for w in weights]


def _active_routes(weights):
    route_names = ["itemcf", "youtube_dnn", "content", "hot_fresh"]
    return [(name, weight) for name, weight in zip(route_names, weights) if weight > 0]


def _fit_recall_models(
    history,
    topk_sim,
    item_embeddings,
    item_category,
    item_popularity,
    item_created_at,
    active_routes,
    youtube_dnn_embedding_dim=128,
    youtube_dnn_epochs=1,
    youtube_dnn_batch_size=256,
):
    recallers = []
    for route_name, _ in active_routes:
        if route_name == "itemcf":
            recallers.append(ItemCF(topk_sim=topk_sim).fit(history))
        elif route_name == "youtube_dnn":
            recallers.append(
                YouTubeDNNRecall(
                    embedding_dim=youtube_dnn_embedding_dim,
                    training_epochs=youtube_dnn_epochs,
                    batch_size=youtube_dnn_batch_size,
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


def _build_item_meta(click_df, articles_df):
    pop_count = click_df["click_article_id"].value_counts()
    max_pop = float(pop_count.max()) if not pop_count.empty else 1.0
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


def run_offline_eval(
    data_dir,
    topk_recall,
    topk_sim,
    popular_fill_k,
    k,
    recall_weights="1,0,0,0",
    youtube_dnn_embedding_dim=128,
    youtube_dnn_epochs=1,
    youtube_dnn_batch_size=256,
    with_ranking=False,
    max_train_users=20000,
):
    train_path = os.path.join(data_dir, "train_click_log.csv")
    train_df = load_click_log(train_path)

    articles_path = os.path.join(data_dir, "articles.csv")
    articles_df = pd.read_csv(articles_path) if os.path.exists(articles_path) else pd.DataFrame()
    emb_path = os.path.join(data_dir, "articles_emb.csv")
    emb_df = pd.read_csv(emb_path) if os.path.exists(emb_path) else pd.DataFrame()

    hist_df, label_df = split_last_click(train_df)
    user_history = build_user_click_history(hist_df)
    hot_items = _popular_items(hist_df, popular_fill_k)

    item_popularity, item_category, item_created_at = _build_item_meta(hist_df, articles_df)

    if emb_df.empty or "article_id" not in emb_df.columns:
        item_embeddings = {}
    else:
        emb_cols = [c for c in emb_df.columns if c != "article_id"]
        item_embeddings = {
            int(row["article_id"]): row[emb_cols].to_numpy(dtype=float)
            for _, row in emb_df.iterrows()
        }

    weights = _parse_weights(recall_weights)
    active_routes = _active_routes(weights)
    active_weights = [w for _, w in active_routes]
    logger.info(
        "Active recall routes: %s",
        ", ".join("{}={:.3f}".format(name, w) for name, w in active_routes),
    )

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
    )

    users = label_df["user_id"].drop_duplicates().tolist()
    if max_train_users and len(users) > max_train_users:
        users = users[:max_train_users]
        logger.info("Offline eval users truncated to %d for ranking", len(users))
    rows = []

    for user_id in users:
        route_results = [{user_id: rec.recall(user_id, topk=topk_recall)} for rec in recallers]
        merged = merge_recall_results(route_results, weights=active_weights).get(user_id, [])
        merged = merged[:topk_recall]

        if len(merged) < topk_recall:
            seen = set(user_history.get(user_id, []))
            existing = {item for item, _ in merged}
            fill_score = -1.0
            for item in hot_items:
                if item in seen or item in existing:
                    continue
                merged.append((item, fill_score))
                fill_score -= 1.0
                if len(merged) >= topk_recall:
                    break

        for article_id, score in merged:
            rows.append((user_id, article_id, score))

    ranked_df = pd.DataFrame(rows, columns=["user_id", "article_id", "rank_score"])

    if with_ranking and not ranked_df.empty:
        candidate_dict = {}
        for user_id, article_id, score in rows:
            candidate_dict.setdefault(int(user_id), []).append((int(article_id), float(score)))

        user_last_ts = _build_user_last_timestamp(hist_df)
        feature_df = build_feature_dataframe(
            candidates_by_user=candidate_dict,
            user_history=user_history,
            item_embeddings=item_embeddings,
            item_category=item_category,
            item_created_at=item_created_at,
            item_popularity=item_popularity,
            user_last_ts=user_last_ts,
        )

        if not feature_df.empty:
            labels = label_df.rename(columns={"click_article_id": "article_id"})[
                ["user_id", "article_id"]
            ].copy()
            labels = labels[labels["user_id"].isin(users)]
            labels["label"] = 1
            ranker = GBDTLRRanker().fit(feature_df, labels)
            ranked_df = ranker.predict(feature_df)
            logger.info("Applied GBDT+LR ranking for offline eval")
        else:
            logger.warning("Ranking skipped: empty feature dataframe")

    eval_labels = label_df[["user_id", "click_article_id"]]
    if max_train_users:
        eval_labels = eval_labels[eval_labels["user_id"].isin(users)]
    metrics = evaluate(ranked_df, eval_labels, k=k)

    logger.info("Offline metrics on train leave-one-out: %s", metrics)
    return metrics


def main():
    args = parse_args()
    run_offline_eval(
        data_dir=args.data_dir,
        topk_recall=args.topk_recall,
        topk_sim=args.topk_sim,
        popular_fill_k=args.popular_fill_k,
        k=args.k,
        recall_weights=args.recall_weights,
        youtube_dnn_embedding_dim=args.youtube_dnn_embedding_dim,
        youtube_dnn_epochs=args.youtube_dnn_epochs,
        youtube_dnn_batch_size=args.youtube_dnn_batch_size,
        with_ranking=args.with_ranking,
        max_train_users=args.max_train_users,
    )


if __name__ == "__main__":
    main()
