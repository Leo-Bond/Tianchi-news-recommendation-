"""Data loading and preprocessing for the Tianchi news recommendation task.

Expected raw files (CSV format, placed under *data_dir*):
  - train_click_log.csv   : historical click log used for training
  - testA_click_log.csv   : click log for test-set users (last click is hidden)
  - articles.csv          : article meta-data (category, created_at, words_count)
  - articles_emb.csv      : pre-computed article embeddings (optional)

Column conventions
------------------
click_log:
  user_id, click_article_id, click_timestamp, click_environment,
  click_deviceGroup, click_os, click_country, click_region,
  click_referrer_type

articles:
  article_id, category_id, created_at_ts, words_count

articles_emb:
  article_id, emb_0, emb_1, …, emb_249
"""

import os
import pandas as pd

try:
  from .utils import get_logger, timer
except ImportError:  # support: python data_processing.py
  from utils import get_logger, timer

logger = get_logger(__name__, source_file=__file__)


@timer
def load_click_log(path):
    """Load a click-log CSV and return a sorted DataFrame."""
    df = pd.read_csv(path)
    df.sort_values(["user_id", "click_timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Loaded click log: %s  shape=%s", path, df.shape)
    return df


@timer
def load_articles(path):
    """Load the article meta-data CSV."""
    df = pd.read_csv(path)
    logger.info("Loaded articles: %s  shape=%s", path, df.shape)
    return df


@timer
def load_article_embeddings(path):
    """Load pre-computed article embeddings CSV."""
    df = pd.read_csv(path)
    logger.info("Loaded embeddings: %s  shape=%s", path, df.shape)
    return df


def build_user_click_history(click_df):
    """Return a mapping {user_id -> list of article_ids} ordered by click time.

    The list is ordered from *earliest* to *latest* click.
    """
    history = (
        click_df.groupby("user_id")["click_article_id"]
        .apply(list)
        .to_dict()
    )
    return history


def split_last_click(click_df):
    """Split click log into history (all but last click) and labels (last click).

    Returns
    -------
    hist_df : pd.DataFrame  – all clicks except the last one per user
    label_df : pd.DataFrame – the last click per user (ground-truth for eval)
    """
    click_df = click_df.sort_values(["user_id", "click_timestamp"])
    last_idx = click_df.groupby("user_id").tail(1).index
    label_df = click_df.loc[last_idx].reset_index(drop=True)
    hist_df = click_df.drop(index=last_idx).reset_index(drop=True)
    return hist_df, label_df


def load_dataset(data_dir):
    """Convenience function: load all raw files from *data_dir*.

    Returns
    -------
    train_df, test_df, articles_df : pd.DataFrame
    """
    train_path = os.path.join(data_dir, "train_click_log.csv")
    test_path = os.path.join(data_dir, "testA_click_log.csv")
    articles_path = os.path.join(data_dir, "articles.csv")

    train_df = load_click_log(train_path)
    test_df = load_click_log(test_path)
    articles_df = load_articles(articles_path)
    return train_df, test_df, articles_df


def main(data_dir=None):
  """Simple smoke test for this module."""
  if data_dir is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "tcdata")

  train_df, test_df, articles_df = load_dataset(data_dir)
  hist_df, label_df = split_last_click(train_df)
  train_history = build_user_click_history(hist_df)

  logger.info("Smoke test passed.")
  logger.info("train_df=%s test_df=%s articles_df=%s", train_df.shape, test_df.shape, articles_df.shape)
  logger.info("hist_df=%s label_df=%s users_in_history=%d", hist_df.shape, label_df.shape, len(train_history))


if __name__ == "__main__":
    main()
