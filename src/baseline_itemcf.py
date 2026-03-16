"""Multi-route recall + GBDT+LR ranking pipeline for Tianchi news recommendation.

Usage
-----
python -m src.baseline_itemcf --data_dir tcdata --output_dir output
"""

import argparse
import os

import pandas as pd

try:
    from .data_processing import build_user_click_history, split_last_click
    from .ranking import GBDTLRRanker, build_feature_dataframe
    from .recall import (
        ContentSimilarityRecall,
        HotFreshRecall,
        ItemCF,
        YouTubeDNNRecall,
        merge_recall_results,
    )
    from .evaluate import make_submission
    from .utils import get_logger
except ImportError:  # support direct script execution
    from data_processing import build_user_click_history, split_last_click
    from ranking import GBDTLRRanker, build_feature_dataframe
    from recall import (
        ContentSimilarityRecall,
        HotFreshRecall,
        ItemCF,
        YouTubeDNNRecall,
        merge_recall_results,
    )
    from evaluate import make_submission
    from utils import get_logger

logger = get_logger(__name__, source_file=__file__)


def parse_args():
    parser = argparse.ArgumentParser(description="ItemCF baseline submission generator")
    parser.add_argument("--data_dir", default="tcdata", help="Directory containing click logs")
    parser.add_argument("--output_dir", default="output", help="Directory to save submission")
    parser.add_argument("--topk_recall", type=int, default=50, help="Recall candidates per user")
    parser.add_argument("--topk_submit", type=int, default=5, help="Final submit items per user")
    parser.add_argument("--topk_sim", type=int, default=20, help="ItemCF top similar items kept per item")
    parser.add_argument("--popular_fill_k", type=int, default=200, help="Hot items pool size for fallback")
    parser.add_argument(
        "--recall_weights",
        default="0.4,0.2,0.2,0.2",
        help="Weights for itemcf,youtube_dnn,content,hot_fresh recalls",
    )
    return parser.parse_args()


def _resolve_test_path(data_dir):
    candidates = [
        "testA_click_log.csv",
        "testB_click_log_Test_B.csv",
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
    if len(weights) != 4:
        return [0.4, 0.2, 0.2, 0.2]
    total = sum(weights)
    if total <= 0:
        return [0.4, 0.2, 0.2, 0.2]
    return [w / total for w in weights]


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
):
    itemcf = ItemCF(topk_sim=topk_sim).fit(history)
    youtube_dnn = YouTubeDNNRecall().fit(history, item_embeddings=item_embeddings)
    content = ContentSimilarityRecall().fit(
        history, item_embeddings=item_embeddings, item_category=item_category
    )
    hot_fresh = HotFreshRecall().fit(
        history,
        item_popularity=item_popularity,
        item_created_at=item_created_at,
    )
    return itemcf, youtube_dnn, content, hot_fresh


def _multi_recall(
    users,
    recallers,
    history,
    topk_recall,
    route_weights,
    popular_items,
):
    results = {}
    for user_id in users:
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
    return results


def build_baseline_submission(
    data_dir,
    output_dir,
    topk_recall,
    topk_submit,
    topk_sim,
    popular_fill_k,
    recall_weights="0.4,0.2,0.2,0.2",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_path = os.path.join(data_dir, "train_click_log.csv")
    test_path = _resolve_test_path(data_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    articles_df = _load_optional_frame(os.path.join(data_dir, "articles.csv"))
    emb_df = _load_optional_frame(os.path.join(data_dir, "articles_emb.csv"))
    logger.info("Loaded train: %s shape=%s", train_path, train_df.shape)
    logger.info("Loaded test: %s shape=%s", test_path, test_df.shape)

    all_click = pd.concat([train_df, test_df], ignore_index=True)
    all_click.sort_values(["user_id", "click_timestamp"], inplace=True)
    item_embeddings = _build_item_embeddings(emb_df)
    item_popularity, item_category, item_created_at = _build_item_meta(all_click, articles_df)
    weights = _parse_weights(recall_weights)
    hot_items = _popular_items(all_click, popular_fill_k)

    # ranker training data: leave-one-out on train clicks
    hist_df, label_df = split_last_click(train_df)
    train_history = _build_history(hist_df)
    train_users = label_df["user_id"].drop_duplicates().tolist()
    train_last_ts = _build_user_last_timestamp(hist_df)
    train_recallers = _fit_recall_models(
        history=train_history,
        topk_sim=topk_sim,
        item_embeddings=item_embeddings,
        item_category=item_category,
        item_popularity=item_popularity,
        item_created_at=item_created_at,
    )
    train_candidates = _multi_recall(
        users=train_users,
        recallers=train_recallers,
        history=train_history,
        topk_recall=topk_recall,
        route_weights=weights,
        popular_items=hot_items,
    )
    train_feature_df = build_feature_dataframe(
        candidates_by_user=train_candidates,
        user_history=train_history,
        item_embeddings=item_embeddings,
        item_category=item_category,
        item_created_at=item_created_at,
        item_popularity=item_popularity,
        user_last_ts=train_last_ts,
    )
    if train_feature_df.empty:
        logger.warning("No train features built; fallback to recall score ranking.")

    train_labels = label_df.rename(columns={"click_article_id": "article_id"})[
        ["user_id", "article_id"]
    ].copy()
    train_labels["label"] = 1

    ranker = GBDTLRRanker().fit(train_feature_df, train_labels) if not train_feature_df.empty else None

    # inference recall on all-click history
    user_history = _build_history(all_click)
    test_users = test_df["user_id"].drop_duplicates().tolist()
    test_last_ts = _build_user_last_timestamp(all_click)
    recallers = _fit_recall_models(
        history=user_history,
        topk_sim=topk_sim,
        item_embeddings=item_embeddings,
        item_category=item_category,
        item_popularity=item_popularity,
        item_created_at=item_created_at,
    )
    test_candidates = _multi_recall(
        users=test_users,
        recallers=recallers,
        history=user_history,
        topk_recall=topk_recall,
        route_weights=weights,
        popular_items=hot_items,
    )
    test_feature_df = build_feature_dataframe(
        candidates_by_user=test_candidates,
        user_history=user_history,
        item_embeddings=item_embeddings,
        item_category=item_category,
        item_created_at=item_created_at,
        item_popularity=item_popularity,
        user_last_ts=test_last_ts,
    )

    if ranker is not None and not test_feature_df.empty:
        ranked_df = ranker.predict(test_feature_df)
    else:
        rows = []
        for user_id, cands in test_candidates.items():
            for article_id, score in cands:
                rows.append((user_id, article_id, score))
        ranked_df = pd.DataFrame(rows, columns=["user_id", "article_id", "rank_score"])

    submission = make_submission(ranked_df, topk=topk_submit)

    submit_path = os.path.join(output_dir, "submission_itemcf_baseline.csv")
    submission.to_csv(submit_path, index=False)
    logger.info("Saved baseline submission to %s shape=%s", submit_path, submission.shape)
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
    )


if __name__ == "__main__":
    main()
