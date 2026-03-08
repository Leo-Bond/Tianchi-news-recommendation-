"""End-to-end pipeline for the Tianchi news recommendation task.

Usage
-----
    python main.py --data_dir data/ --output_dir output/ --mode train
    python main.py --data_dir data/ --output_dir output/ --mode predict

Modes
-----
train   : Build recall, train LightGBM ranker, evaluate on held-out split.
predict : Generate submission file for the test set.
"""

import argparse
import os

import pandas as pd

from src.data_processing import load_dataset, split_last_click, build_user_click_history
from src.feature_engineering import build_user_features, build_candidate_features
from src.recall import ItemCF, UserCF, merge_recall_results
from src.rank import LGBRanker, build_training_samples
from src.evaluate import evaluate, make_submission
from src.utils import get_logger, save_pickle, load_pickle

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Tianchi news recommendation pipeline")
    parser.add_argument("--data_dir", default="data/", help="Directory containing raw CSV files")
    parser.add_argument("--output_dir", default="output/", help="Directory for artefacts and submissions")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--topk_recall", type=int, default=50, help="Recall candidates per user")
    parser.add_argument("--topk_submit", type=int, default=5, help="Articles per user in submission")
    parser.add_argument("--n_epochs_bpr", type=int, default=0, help="BPR epochs (0 = skip BPR)")
    return parser.parse_args()


def run_recall(click_history: dict, topk: int) -> dict:
    """Run ItemCF + UserCF and merge the results."""
    item_cf = ItemCF(topk_sim=20).fit(click_history)
    user_cf = UserCF(topk_users=20).fit(click_history)

    item_cf_results = {uid: item_cf.recall(uid, topk) for uid in click_history}
    user_cf_results = {uid: user_cf.recall(uid, topk) for uid in click_history}

    merged = merge_recall_results([item_cf_results, user_cf_results], weights=[0.7, 0.3])
    return merged


def train_pipeline(args):
    logger.info("=== TRAIN MODE ===")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    train_df, _test_df, articles_df = load_dataset(args.data_dir)

    # 2. Split last click for evaluation
    hist_df, label_df = split_last_click(train_df)
    click_history = build_user_click_history(hist_df)

    # 3. Recall
    logger.info("Running recall …")
    recall_results = run_recall(click_history, topk=args.topk_recall)
    save_pickle(recall_results, os.path.join(args.output_dir, "recall_results.pkl"))

    # 4. Feature engineering
    user_features = build_user_features(hist_df, articles_df)
    feat_df = build_candidate_features(recall_results, user_features, click_history, articles_df)

    # 5. Build training samples with labels
    train_samples = build_training_samples(feat_df, label_df)

    # 6. Train LightGBM ranker
    ranker = LGBRanker(n_estimators=200)
    ranker.fit(train_samples)
    save_pickle(ranker, os.path.join(args.output_dir, "ranker.pkl"))

    # 7. Evaluate on training users (self-check; use cross-val for real evaluation)
    ranked = ranker.rank(feat_df, topk=args.topk_submit)
    metrics = evaluate(ranked, label_df, k=args.topk_submit)
    logger.info("Metrics: %s", metrics)

    return metrics


def predict_pipeline(args):
    logger.info("=== PREDICT MODE ===")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    train_df, test_df, articles_df = load_dataset(args.data_dir)
    train_history = build_user_click_history(train_df)
    test_history = build_user_click_history(test_df)

    # Merge histories so the recall model sees all clicks
    combined_history = {**train_history}
    for uid, items in test_history.items():
        if uid in combined_history:
            combined_history[uid] = combined_history[uid] + items
        else:
            combined_history[uid] = items

    # 2. Recall for test users
    test_recall = run_recall(
        {uid: combined_history[uid] for uid in test_history},
        topk=args.topk_recall,
    )

    # 3. Features
    user_features = build_user_features(train_df, articles_df)
    feat_df = build_candidate_features(test_recall, user_features, combined_history, articles_df)

    # 4. Rank
    ranker_path = os.path.join(args.output_dir, "ranker.pkl")
    if not os.path.exists(ranker_path):
        raise FileNotFoundError(f"Trained ranker not found at {ranker_path}. Run with --mode train first.")
    ranker: LGBRanker = load_pickle(ranker_path)
    ranked = ranker.rank(feat_df, topk=args.topk_submit)

    # 5. Submission
    submission = make_submission(ranked, topk=args.topk_submit)
    submit_path = os.path.join(args.output_dir, "submission.csv")
    submission.to_csv(submit_path, index=False)
    logger.info("Submission saved to %s  shape=%s", submit_path, submission.shape)

    return submission


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train_pipeline(args)
    else:
        predict_pipeline(args)
