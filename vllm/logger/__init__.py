"""Logging configuration for vLLM."""
import datetime
import logging
import os
import sys
from functools import partial
from logging import Logger
from logging import config as logging_config
from typing import Optional

VLLM_CONFIGURE_LOGGING = int(os.getenv("VLLM_CONFIGURE_LOGGING", "1"))

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "vllm": {
            "class": "vllm.logging.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "vllm": {
            "class": "logging.StreamHandler",
            "formatter": "vllm",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "vllm": {
            "handlers": ["vllm"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "version": 1,
}


def _configure_vllm_root_logger() -> None:
    logging_config.dictConfig(DEFAULT_LOGGING_CONFIG)


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
if VLLM_CONFIGURE_LOGGING:
    _configure_vllm_root_logger()


def _configure_vllm_logger(logger: Logger) -> None:
    # Use the same settings as for root logger
    _root_logger = logging.getLogger("vllm")
    default_log_level = os.getenv("LOG_LEVEL", _root_logger.level)
    logger.setLevel(default_log_level)
    for handler in _root_logger.handlers:
        logger.addHandler(handler)
    logger.propagate = _root_logger.propagate


def init_logger(name: str) -> Logger:
    logger = logging.getLogger(name)
    if VLLM_CONFIGURE_LOGGING:
        _configure_vllm_logger(logger)
    return logger


logger = init_logger(__name__)


def _trace_calls(log_path, root_dir, frame, event, arg=None):
    if event in ['call', 'return']:
        # Extract the filename, line number, function name, and the code object
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        if not filename.startswith(root_dir):
            # only log the functions in the vllm root_dir
            return
        # Log every function call or return
        try:
            with open(log_path, 'a') as f:
                if event == 'call':
                    f.write(f"{datetime.datetime.now()} Call to"
                            f" {func_name} in {filename}:{lineno}\n")
                else:
                    f.write(f"{datetime.datetime.now()} Return from"
                            f" {func_name} in {filename}:{lineno}\n")
        except NameError:
            # modules are deleted during shutdown
            pass
    return partial(_trace_calls, log_path, root_dir)


def enable_trace_function_call(log_file_path: str,
                               root_dir: Optional[str] = None):
    """
    Enable tracing of every function call in code under `root_dir`.
    This is useful for debugging hangs or crashes.
    `log_file_path` is the path to the log file.
    `root_dir` is the root directory of the code to trace. If None, it is the
    vllm root directory.

    Note that this call is thread-level, any threads calling this function
    will have the trace enabled. Other threads will not be affected.
    """
    logger.warning(
        "VLLM_TRACE_FUNCTION is enabled. It will record every"
        " function executed by Python. This will slow down the code. It "
        "is suggested to be used for debugging hang or crashes only.")
    logger.info(f"Trace frame log is saved to {log_file_path}")
    if root_dir is None:
        # by default, this is the vllm root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))
