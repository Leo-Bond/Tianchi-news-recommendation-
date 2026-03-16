# Tianchi News Recommendation

A clean, baseline-first framework for the [Tianchi News Recommendation competition](https://tianchi.aliyun.com/competition/entrance/531842/introduction).

## Overview

The task is to predict the next article a user will click based on historical behaviour.
This repo now supports a **multi-route recall + GBDT+LR ranking** framework:

- Recall routes: ItemCF, YouTubeDNN (DeepMatch+DeepCTR+TensorFlow, with proxy fallback), content similarity, hot/fresh
- Ranking features: recall score, user-item embedding similarity, category match, publish time gap, article popularity, user recent interest distribution
- Ranker: GBDT leaf features + LR final scoring (with deterministic fallback when sklearn is unavailable)

```
Raw data  ──►  Multi-route recall merge  ──►  Feature engineering  ──►  GBDT+LR ranking  ──►  Submission
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── main.py                # Multi-route recall + ranking entrypoint
│   ├── multi_recall_ranking.py # Multi-route recall + GBDT+LR pipeline
│   ├── baseline_itemcf.py     # ItemCF-only baseline pipeline
│   ├── utils.py               # Logging, timing, pickle helpers
│   ├── data_processing.py     # Load CSVs, split history / label
│   ├── recall.py              # ItemCF implementation (+ optional UserCF/BPR)
│   └── evaluate.py            # Submission export and metrics helpers
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Optional (enable DeepMatch YouTubeDNN backend):

```bash
pip install -r requirements-deepmatch.txt
```

### 2. Prepare data

Place the competition data under `tcdata/` (or your custom `--data_dir`):

| File | Description |
|------|-------------|
| `tcdata/train_click_log.csv` | Training user click logs |
| `tcdata/testA_click_log.csv` | Test user click logs |
| `tcdata/articles.csv` | Article meta-data (optional for baseline) |

### 3. Run baseline submission

```bash
python -m src.main --data_dir tcdata --output_dir output
```

Saves `output/submission_multi_recall_ranking.csv`.

Run ItemCF-only baseline:

```bash
python -m src.baseline_itemcf --data_dir tcdata --output_dir output
```

Saves `output/submission_itemcf_baseline.csv`.

### 4. Offline test on train (leave-one-out)

```bash
python -m src.offline_eval --data_dir tcdata --k 5
```

This uses `train_click_log.csv` only: for each user, the last click is treated as label and the earlier clicks are used as history.

## Module Reference

### `src.recall`

| Class | Description |
|-------|-------------|
| `ItemCF` | Item-based CF with IUF weighting and positional decay |
| `UserCF` | User-based CF with IIF weighting |
| `BPR` | Bayesian Personalised Ranking (SGD, latent factor model) |
| `merge_recall_results()` | Weighted merge of multiple recall dicts |

### `src.evaluate`

| Function | Description |
|----------|-------------|
| `evaluate()` | MRR@k, Hit@k, NDCG@k across all users |
| `make_submission()` | Wide-format submission DataFrame |

## CLI Options

```
python -m src.main [-h] [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
               [--topk_recall TOPK_RECALL]
               [--topk_submit TOPK_SUBMIT]
               [--topk_sim TOPK_SIM]
               [--popular_fill_k POPULAR_FILL_K]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `tcdata` | Directory with raw CSV files |
| `--output_dir` | `output/` | Output artefacts directory |
| `--topk_recall` | `50` | Recall candidates per user |
| `--topk_submit` | `5` | Articles per user in submission |
| `--topk_sim` | `20` | Similar items kept per clicked item |
| `--popular_fill_k` | `200` | Hot-item pool size for recall fallback |
| `--recall_weights` | `0.4,0.2,0.2,0.2` | Weights for ItemCF/YouTubeDNN/content/hot-fresh recall merge |
| `--youtube_dnn_use_deepmatch` | `False` | Use DeepMatch+DeepCTR+TensorFlow backend for YouTubeDNN |
| `--youtube_dnn_embedding_dim` | `16` | Embedding size for DeepMatch YouTubeDNN |
| `--youtube_dnn_epochs` | `1` | Training epochs for DeepMatch YouTubeDNN |
| `--youtube_dnn_batch_size` | `256` | Training batch size for DeepMatch YouTubeDNN |

## License

MIT
