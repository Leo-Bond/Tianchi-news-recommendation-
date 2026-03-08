"""Baseline-first entrypoint for Tianchi news recommendation.

Usage
-----
python -m src.main --data_dir tcdata --output_dir output
"""

import argparse

from src.baseline_itemcf import build_baseline_submission


def parse_args():
    parser = argparse.ArgumentParser(description="ItemCF baseline pipeline")
    parser.add_argument("--data_dir", default="tcdata", help="Directory containing click logs")
    parser.add_argument("--output_dir", default="output", help="Directory to save submission")
    parser.add_argument("--topk_recall", type=int, default=50, help="Recall candidates per user")
    parser.add_argument("--topk_submit", type=int, default=5, help="Final submit items per user")
    parser.add_argument("--topk_sim", type=int, default=20, help="ItemCF top similar items kept per item")
    parser.add_argument("--popular_fill_k", type=int, default=200, help="Hot items pool size for fallback")
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
    )


if __name__ == "__main__":
    main()
