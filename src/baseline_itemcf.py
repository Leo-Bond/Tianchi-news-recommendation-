"""Simple ItemCF baseline for Tianchi news recommendation.

Usage
-----
python -m src.baseline_itemcf --data_dir tcdata --output_dir output
"""

import argparse
import os

import pandas as pd

try:
    from .recall import ItemCF
    from .evaluate import make_submission
    from .utils import get_logger
except ImportError:  # support direct script execution
    from recall import ItemCF
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
    return (
        click_df.sort_values(["user_id", "click_timestamp"])
        .groupby("user_id")["click_article_id"]
        .apply(list)
        .to_dict()
    )


def _popular_items(click_df, k):
    return click_df["click_article_id"].value_counts().head(k).index.tolist()


def _recall_for_user(
    user_id,
    itemcf,
    history,
    topk_recall,
    popular_items,
):
    recalled = itemcf.recall(user_id, topk=topk_recall)
    if len(recalled) >= topk_recall:
        return recalled

    seen_items = set(history.get(user_id, []))
    existing_items = {item for item, _ in recalled}
    filled = list(recalled)

    fill_score = -1.0
    for item in popular_items:
        if item in seen_items or item in existing_items:
            continue
        filled.append((item, fill_score))
        fill_score -= 1.0
        if len(filled) >= topk_recall:
            break

    return filled


def build_baseline_submission(
    data_dir,
    output_dir,
    topk_recall,
    topk_submit,
    topk_sim,
    popular_fill_k,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_path = os.path.join(data_dir, "train_click_log.csv")
    test_path = _resolve_test_path(data_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logger.info("Loaded train: %s shape=%s", train_path, train_df.shape)
    logger.info("Loaded test: %s shape=%s", test_path, test_df.shape)

    all_click = pd.concat([train_df, test_df], ignore_index=True)
    all_click.sort_values(["user_id", "click_timestamp"], inplace=True)

    user_history = _build_history(all_click)
    test_users = test_df["user_id"].drop_duplicates().tolist()
    hot_items = _popular_items(all_click, popular_fill_k)

    itemcf = ItemCF(topk_sim=topk_sim).fit(user_history)

    rows = []
    for user_id in test_users:
        candidates = _recall_for_user(
            user_id=user_id,
            itemcf=itemcf,
            history=user_history,
            topk_recall=topk_recall,
            popular_items=hot_items,
        )
        for article_id, score in candidates:
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
    )


if __name__ == "__main__":
    main()
