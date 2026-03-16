"""Utility functions for logging, timing, and I/O."""

import os
import time
import pickle
import logging
import inspect
import numpy as np
from functools import wraps


def _resolve_log_stem(name, source_file=None):
    if source_file:
        return os.path.splitext(os.path.basename(source_file))[0]

    if name and name != "__main__":
        return name.split(".")[-1]

    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        caller_file = frame.f_back.f_globals.get("__file__")
        if caller_file:
            return os.path.splitext(os.path.basename(caller_file))[0]

    return "app"


def _default_log_file(name, source_file=None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "output", "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_stem = _resolve_log_stem(name, source_file=source_file)
    return os.path.join(log_dir, "{}.log".format(log_stem))


def _has_stream_handler(logger):
    return any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)


def _has_file_handler(logger, file_path):
    target = os.path.abspath(file_path)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(getattr(handler, "baseFilename", "")) == target:
                return True
    return False


def get_logger(name, level=logging.INFO, source_file=None):
    """Return a logger that writes to stdout and output/log/<module>.log."""
    if source_file is None and (not name or name == "__main__"):
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            source_file = frame.f_back.f_globals.get("__file__")

    logger = logging.getLogger(name)

    fmt = logging.Formatter(
        "%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not _has_stream_handler(logger):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    log_file_path = _default_log_file(name, source_file=source_file)
    if not _has_file_handler(logger, log_file_path):
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False

    logger.setLevel(level)
    return logger


def timer(func):
    """Decorator that logs the wall-clock time of a function call."""
    logger = get_logger(func.__module__, source_file=func.__code__.co_filename)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info("%s finished in %.2fs", func.__name__, elapsed)
        return result

    return wrapper


def save_pickle(obj, path):
    """Pickle *obj* to *path*, creating parent directories as needed."""
    parent_dir = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """Load and return a pickled object from *path*."""
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_normalize(vec):
    """Return l2-normalized vector while keeping zero vectors unchanged."""
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm
