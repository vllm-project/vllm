import logging
import os
import sys
import tempfile

from vllm.logger import (_DATE_FORMAT, _FORMAT, enable_trace_function_call,
                         init_logger)
from vllm.logging import NewLineFormatter


def f1(x):
    return f2(x)


def f2(x):
    return x


def test_trace_function_call():
    fd, path = tempfile.mkstemp()
    cur_dir = os.path.dirname(__file__)
    enable_trace_function_call(path, cur_dir)
    f1(1)
    with open(path, 'r') as f:
        content = f.read()

    assert "f1" in content
    assert "f2" in content
    sys.settrace(None)
    os.remove(path)


def test_vllm_root_logger_configuration():
    logger = logging.getLogger("vllm")
    assert logger.level == logging.DEBUG
    assert not logger.propagate

    handler = logger.handlers[0]
    assert handler.stream == sys.stdout
    assert handler.level == logging.INFO

    formatter = handler.formatter
    assert formatter is not None
    assert isinstance(formatter, NewLineFormatter)
    assert formatter._fmt == _FORMAT
    assert formatter.datefmt == _DATE_FORMAT


def test_init_logger_configures_the_logger_like_the_root_logger():
    root_logger = logging.getLogger("vllm")
    logger = init_logger(__name__)

    assert logger.name == __name__
    assert logger.level == logging.DEBUG
    assert logger.handlers == root_logger.handlers
    assert logger.propagate == root_logger.propagate
