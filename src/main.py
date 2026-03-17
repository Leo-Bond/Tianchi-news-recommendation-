"""Baseline-first entrypoint for Tianchi news recommendation.

Usage
-----
python -m src.main --data_dir tcdata --output_dir output
"""

import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from src.multi_recall_ranking import build_baseline_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-route recall + GBDT+LR ranking pipeline")
    parser.add_argument("--data_dir", default="tcdata", help="Directory containing click logs")
    parser.add_argument("--output_dir", default="output", help="Directory to save submission")
    parser.add_argument(
        "--topk_recall",
        type=int,
        default=10,
        help="Recall candidates per user. Default is lightweight to reduce runtime and memory.",
    )
    parser.add_argument("--topk_submit", type=int, default=5, help="Final submit items per user")
    parser.add_argument(
        "--topk_sim",
        type=int,
        default=10,
        help="ItemCF top similar items kept per item. Default is lightweight to reduce runtime.",
    )
    parser.add_argument(
        "--popular_fill_k",
        type=int,
        default=100,
        help="Hot items pool size for fallback. Default is lightweight to reduce runtime.",
    )
    parser.add_argument(
        "--recall_weights",
        default="1,0,0,0",
        help="Weights for itemcf,youtube_dnn,content,hot_fresh recalls. Default is lightweight ItemCF-only.",
    )
    parser.add_argument(
        "--max_train_users",
        type=int,
        default=20000,
        help="Maximum number of training users used to build ranker candidates. Default is lightweight.",
    )
    parser.add_argument("--youtube_dnn_embedding_dim", type=int, default=128, help="Hidden size for the PyTorch YouTubeDNN user tower")
    parser.add_argument("--youtube_dnn_epochs", type=int, default=1, help="Training epochs for the PyTorch YouTubeDNN")
    parser.add_argument("--youtube_dnn_batch_size", type=int, default=256, help="Batch size for the PyTorch YouTubeDNN")
    return parser.parse_args()


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
    )


if __name__ == "__main__":
    main()
