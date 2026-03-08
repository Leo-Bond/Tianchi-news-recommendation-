"""Utility functions for logging, timing, and I/O."""

import os
import time
import pickle
import logging
from functools import wraps


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout with a standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(name)s  %(levelname)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def timer(func):
    """Decorator that logs the wall-clock time of a function call."""
    logger = get_logger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info("%s finished in %.2fs", func.__name__, elapsed)
        return result

    return wrapper


def save_pickle(obj, path: str) -> None:
    """Pickle *obj* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    """Load and return a pickled object from *path*."""
    with open(path, "rb") as f:
        return pickle.load(f)
