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
    from .recall import ItemCF
    from .evaluate import evaluate
    from .utils import get_logger
except ImportError:
    from data_processing import load_click_log, split_last_click, build_user_click_history
    from recall import ItemCF
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
    return parser.parse_args()


def _popular_items(click_df, k):
    return click_df["click_article_id"].value_counts().head(k).index.tolist()


def _recall_with_fallback(itemcf, user_id, user_history, topk_recall, hot_items):
    recalled = itemcf.recall(user_id, topk=topk_recall)
    if len(recalled) >= topk_recall:
        return recalled

    seen = set(user_history.get(user_id, []))
    existing = set([item for item, _ in recalled])
    filled = list(recalled)
    fill_score = -1.0

    for item in hot_items:
        if item in seen or item in existing:
            continue
        filled.append((item, fill_score))
        fill_score -= 1.0
        if len(filled) >= topk_recall:
            break

    return filled


def run_offline_eval(data_dir, topk_recall, topk_sim, popular_fill_k, k):
    train_path = os.path.join(data_dir, "train_click_log.csv")
    train_df = load_click_log(train_path)

    hist_df, label_df = split_last_click(train_df)
    user_history = build_user_click_history(hist_df)
    hot_items = _popular_items(hist_df, popular_fill_k)

    itemcf = ItemCF(topk_sim=topk_sim).fit(user_history)

    users = label_df["user_id"].drop_duplicates().tolist()
    rows = []

    for user_id in users:
        candidates = _recall_with_fallback(
            itemcf=itemcf,
            user_id=user_id,
            user_history=user_history,
            topk_recall=topk_recall,
            hot_items=hot_items,
        )
        for article_id, score in candidates:
            rows.append((user_id, article_id, score))

    ranked_df = pd.DataFrame(rows, columns=["user_id", "article_id", "rank_score"])
    metrics = evaluate(ranked_df, label_df[["user_id", "click_article_id"]], k=k)

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
    )


if __name__ == "__main__":
    main()
