# Tianchi News Recommendation

Solution for the Tianchi News Recommendation task.

Offline result: {'mrr@5': 0.2613946666666666, 'hit_rate@5': 0.4078, 'ndcg@5': 0.29785336858597583}

Online result: 'mrr@5' = 0.269

## Overview

This repository implements a full pipeline:

1. Multi-route candidate recall
2. Feature engineering
3. Learning-to-rank scoring
4. Submission export

Current recall routes:

- ItemCF
- YouTubeDNN recall (PyTorch user tower + item vectors)
- Content similarity recall
- Hot/Fresh recall
- Word2Vec recall
- Bipartite network recall

Ranking model behavior:

- First choice: LightGBM LambdaRank
- Fallback: sklearn GBDT+LR
- Final fallback: deterministic linear score

## Project Structure

```text
.
├── src/
│   ├── main.py
│   ├── multi_recall_ranking.py
│   ├── offline_eval.py
│   ├── ranking.py
│   ├── recall.py
│   ├── evaluate.py
│   ├── baseline_itemcf.py
│   ├── data_processing.py
│   └── utils.py
├── scripts/
│   └── run_youtubednn_recall.py
├── tcdata/
├── output/
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Data

Put data under `tcdata/` (or pass another `--data_dir`):

- `train_click_log.csv`
- `testA_click_log.csv` 
- `articles.csv` 
- `articles_emb.csv` 

## Quick Start

### 1. Generate submission (main entry)

```bash
python -m src.main --data_dir tcdata --output_dir output
```

Output:

- `output/submission_multi_recall_ranking.csv`

Note:

- `src.main` already includes ranking by default. There is no `--with_ranking` flag in `src.main`.

### 2. ItemCF-only baseline

```bash
python -m src.baseline_itemcf --data_dir tcdata --output_dir output
```

Output:

- `output/submission_itemcf_baseline.csv`

### 3. Offline leave-one-out evaluation

Without ranking:

```bash
python -m src.offline_eval --data_dir tcdata
```

With ranking:

```bash
python -m src.offline_eval --data_dir tcdata --with_ranking
```

### 4. Run YouTubeDNN recall only

```bash
python scripts/run_youtubednn_recall.py --data_dir tcdata --topk 10
```

## Recall Weights

`--recall_weights` supports 6 values.

Route order is fixed as:

1. itemcf
2. youtube_dnn
3. content
4. hot_fresh
5. w2v
6. bipartite

Examples:

- `1,0,0,0,0,0` means ItemCF only.
- `1,0,0,0,0,1` means ItemCF + Bipartite.

The code normalizes weights internally.

## Main CLI Options

Command:

```bash
python -m src.main --help
```

Important options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `tcdata` | Input data directory |
| `--output_dir` | `output` | Output directory |
| `--topk_recall` | `10` | Recall candidates per user |
| `--topk_submit` | `5` | Final articles per user in submission |
| `--topk_sim` | `10` | Top similar items kept per item for graph-based recall |
| `--popular_fill_k` | `100` | Hot items pool used for recall fill |
| `--recall_weights` | `1,0,0,0` | Multi-route recall weights (4/5/6 values) |
| `--max_train_users` | `0` | Train-user cap for ranker data (`0` means all) |
| `--youtube_dnn_embedding_dim` | `128` | YouTubeDNN hidden size |
| `--youtube_dnn_epochs` | `1` | YouTubeDNN training epochs |
| `--youtube_dnn_batch_size` | `256` | YouTubeDNN batch size |
| `--youtube_dnn_faiss_ivf_nlist` | `4096` | IVF nlist for FAISS index |
| `--youtube_dnn_faiss_ivf_nprobe` | `32` | IVF nprobe for FAISS search |
| `--youtube_dnn_faiss_ivf_min_items` | `20000` | Minimum item count to enable IVF over FlatIP |

## Offline Eval CLI Options

Command:

```bash
python -m src.offline_eval --help
```

Key differences from main:

- Supports `--with_ranking`.
- Defaults to `--topk_recall 10`.
- Uses only `train_click_log.csv` with leave-one-out labels.

## Logs

Runtime logs are written to `output/log/`, for example:

- `output/log/multi_recall_ranking.log`
- `output/log/offline_eval.log`
- `output/log/recall.log`

## License

MIT
