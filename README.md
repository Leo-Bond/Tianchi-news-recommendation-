# Tianchi News Recommendation

A clean, baseline-first framework for the [Tianchi News Recommendation competition](https://tianchi.aliyun.com/competition/entrance/531842/introduction).

## Overview

The task is to predict the next article a user will click based on historical behaviour.
This repo now focuses on a simple **ItemCF recall baseline** that directly produces a submission file.

```
Raw data  ──►  User history build  ──►  ItemCF recall  ──►  Hot-item fill  ──►  Submission
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── main.py                # Unified baseline entrypoint
│   ├── baseline_itemcf.py     # ItemCF baseline pipeline
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

## License

MIT
