import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import build_user_click_history
from src.recall import YouTubeDNNRecall
from src.utils import get_logger

logger = get_logger(__name__, source_file=__file__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PyTorch YouTubeDNN recall model only")
    parser.add_argument("--data_dir", default="tcdata", help="Directory containing click logs and embeddings")
    parser.add_argument(
        "--history_source",
        choices=["train", "all"],
        default="train",
        help="Use only train clicks or train+test clicks to build user history",
    )
    parser.add_argument("--user_id", type=int, default=None, help="User id to inspect")
    parser.add_argument("--topk", type=int, default=10, help="Number of recalled items to print")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Hidden size of the user tower")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional row limit for quick smoke tests",
    )
    return parser.parse_args()


def load_item_embeddings(path):
    emb_df = pd.read_csv(path)
    emb_cols = [col for col in emb_df.columns if col != "article_id"]
    return {
        int(row["article_id"]): row[emb_cols].to_numpy(dtype=float)
        for _, row in emb_df.iterrows()
    }


def main():
    args = parse_args()

    train_path = os.path.join(args.data_dir, "train_click_log.csv")
    test_path = os.path.join(args.data_dir, "testA_click_log.csv")
    emb_path = os.path.join(args.data_dir, "articles_emb.csv")

    train_df = pd.read_csv(train_path)
    if args.max_rows:
        train_df = train_df.head(args.max_rows).copy()

    frames = [train_df]
    if args.history_source == "all" and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        if args.max_rows:
            test_df = test_df.head(args.max_rows).copy()
        frames.append(test_df)

    click_df = pd.concat(frames, ignore_index=True)
    click_df.sort_values(["user_id", "click_timestamp"], inplace=True)
    history = build_user_click_history(click_df)
    item_embeddings = load_item_embeddings(emb_path)

    model = YouTubeDNNRecall(
        embedding_dim=args.embedding_dim,
        training_epochs=args.epochs,
        batch_size=args.batch_size,
    ).fit(history, item_embeddings)

    if args.user_id is None:
        args.user_id = next(iter(history))

    recalled = model.recall(args.user_id, topk=args.topk)

    logger.info("History source: %s", args.history_source)
    logger.info("Users in history: %d", len(history))
    logger.info("Items with embeddings: %d", len(item_embeddings))
    logger.info("Model backend: %s", model._backend)
    logger.info("Inspect user_id: %s", args.user_id)

    print("Top-{} recall for user {}:".format(args.topk, args.user_id))
    for rank, (article_id, score) in enumerate(recalled, start=1):
        print("{:02d}. article_id={} score={:.6f}".format(rank, article_id, score))


if __name__ == "__main__":
    main()
