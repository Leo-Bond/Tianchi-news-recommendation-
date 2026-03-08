"""Tests for utility functions."""
import os
import tempfile
import logging

import pytest

from src.utils import get_logger, save_pickle, load_pickle


def test_get_logger_returns_logger():
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_get_logger_idempotent():
    """Calling get_logger twice should not add duplicate handlers."""
    logger1 = get_logger("idempotent_test")
    handler_count = len(logger1.handlers)
    logger2 = get_logger("idempotent_test")
    assert len(logger2.handlers) == handler_count


def test_save_and_load_pickle():
    data = {"key": [1, 2, 3], "value": "hello"}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "subdir", "test.pkl")
        save_pickle(data, path)
        loaded = load_pickle(path)
    assert loaded == data


def test_timer_decorator_returns_result():
    from src.utils import timer

    @timer
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
