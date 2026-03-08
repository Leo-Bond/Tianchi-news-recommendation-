# Tianchi News Recommendation

A clean, modular baseline framework for the [Tianchi News Recommendation competition](https://tianchi.aliyun.com/competition/entrance/531842/introduction).

## Overview

The task is to predict the next article a user will click based on their historical reading behaviour.  
The official metric is **MRR@5** (Mean Reciprocal Rank at top-5).

```
Raw data  ──►  Data processing  ──►  Recall  ──►  Feature engineering  ──►  LightGBM rank  ──►  Submission
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── utils.py               # Logging, timing, pickle helpers
│   ├── data_processing.py     # Load CSVs, split history / label
│   ├── feature_engineering.py # User & candidate feature extraction
│   ├── recall.py              # ItemCF, UserCF, BPR, ensemble merge
│   ├── rank.py                # LightGBM ranker + training-sample builder
│   └── evaluate.py            # MRR@k, Hit@k, NDCG@k, submission export
├── tests/                     # pytest unit tests (31 tests)
├── main.py                    # End-to-end CLI pipeline
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Download the competition data and place the files under `data/`:

| File | Description |
|------|-------------|
| `data/train_click_log.csv` | Training user click logs |
| `data/testA_click_log.csv` | Test user click logs |
| `data/articles.csv` | Article meta-data (category, word count, ...) |

### 3. Train

```bash
python main.py --data_dir data/ --output_dir output/ --mode train
```

This will:
1. Load training data and split off the last click per user as a label.
2. Run **ItemCF** and **UserCF** recall (top-50 candidates per user).
3. Build user-level and candidate-level features.
4. Train a **LightGBM** binary classifier (click / no-click).
5. Evaluate MRR@5, Hit@5, NDCG@5 on the held-out last-click split.
6. Save the trained ranker to `output/ranker.pkl`.

### 4. Generate submission

```bash
python main.py --data_dir data/ --output_dir output/ --mode predict
```

Saves `output/submission.csv` in the competition format.

## Module Reference

### `src.recall`

| Class | Description |
|-------|-------------|
| `ItemCF` | Item-based CF with IUF weighting and positional decay |
| `UserCF` | User-based CF with IIF weighting |
| `BPR` | Bayesian Personalised Ranking (SGD, latent factor model) |
| `merge_recall_results()` | Weighted merge of multiple recall dicts |

### `src.rank`

| Symbol | Description |
|--------|-------------|
| `LGBRanker` | Pointwise LightGBM classifier used as a ranker |
| `build_training_samples()` | Attach labels + down-sample negatives |

### `src.evaluate`

| Function | Description |
|----------|-------------|
| `evaluate()` | MRR@k, Hit@k, NDCG@k across all users |
| `make_submission()` | Wide-format submission DataFrame |

## Running Tests

```bash
python -m pytest tests/ -v
```

## CLI Options

```
python main.py [-h] [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
               [--mode {train,predict}]
               [--topk_recall TOPK_RECALL]
               [--topk_submit TOPK_SUBMIT]
               [--n_epochs_bpr N_EPOCHS_BPR]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data/` | Directory with raw CSV files |
| `--output_dir` | `output/` | Output artefacts directory |
| `--mode` | `train` | `train` or `predict` |
| `--topk_recall` | `50` | Recall candidates per user |
| `--topk_submit` | `5` | Articles per user in submission |
| `--n_epochs_bpr` | `0` | BPR training epochs (0 = skip) |

## License

MIT
