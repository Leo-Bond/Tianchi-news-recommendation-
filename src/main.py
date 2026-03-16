"""Baseline-first entrypoint for Tianchi news recommendation.

Usage
-----
python -m src.main --data_dir tcdata --output_dir output
"""

import argparse

from src.multi_recall_ranking import build_baseline_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-route recall + GBDT+LR ranking pipeline")
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
    parser.add_argument("--youtube_dnn_use_deepmatch", action="store_true", help="Use DeepMatch/DeepCTR/TensorFlow for YouTubeDNN recall when available")
    parser.add_argument("--youtube_dnn_embedding_dim", type=int, default=16, help="Embedding dim for DeepMatch YouTubeDNN")
    parser.add_argument("--youtube_dnn_epochs", type=int, default=1, help="Training epochs for DeepMatch YouTubeDNN")
    parser.add_argument("--youtube_dnn_batch_size", type=int, default=256, help="Batch size for DeepMatch YouTubeDNN")
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
        youtube_dnn_use_deepmatch=args.youtube_dnn_use_deepmatch,
        youtube_dnn_embedding_dim=args.youtube_dnn_embedding_dim,
        youtube_dnn_epochs=args.youtube_dnn_epochs,
        youtube_dnn_batch_size=args.youtube_dnn_batch_size,
    )


if __name__ == "__main__":
    main()
