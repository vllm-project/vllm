# SPDX-License-Identifier: Apache-2.0
"""Logging configuration for vLLM."""
import datetime
import json
import logging
import os
import sys
from functools import lru_cache, partial
from logging import Logger
from logging.config import dictConfig
from os import path
from types import MethodType
from typing import Any, Dict, Optional, cast

import vllm.envs as envs

VLLM_CONFIGURE_LOGGING = envs.VLLM_CONFIGURE_LOGGING
VLLM_LOGGING_CONFIG_PATH = envs.VLLM_LOGGING_CONFIG_PATH
VLLM_LOGGING_LEVEL = envs.VLLM_LOGGING_LEVEL
VLLM_LOGGING_PREFIX = envs.VLLM_LOGGING_PREFIX
VLLM_LOG_FILTER_PATTERNS = envs.VLLM_LOG_FILTER_PATTERNS

_FORMAT = (f"{VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "
           "%(filename)s:%(lineno)d] %(message)s")
_DATE_FORMAT = "%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    "formatters": {
        "vllm": {
            "class": "vllm.logging_utils.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    # Custom filter uses () to specify its callable
    # https://docs.python.org/3/howto/logging-cookbook.html#configuring-filters-with-dictconfig # noqa: E501
    "filters": {
        "vllm_redact": {
            "()": "vllm.logging_utils.RedactFilter",
            "patterns": VLLM_LOG_FILTER_PATTERNS,
        },
    },
    "handlers": {
        "vllm": {
            "class": "logging.StreamHandler",
            "formatter": "vllm",
            "filters": ["vllm_redact"],
            "level": VLLM_LOGGING_LEVEL,
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
    "disable_existing_loggers": False
}


@lru_cache
def _print_info_once(logger: Logger, msg: str) -> None:
    # Set the stacklevel to 2 to print the original caller's line info
    logger.info(msg, stacklevel=2)


@lru_cache
def _print_warning_once(logger: Logger, msg: str) -> None:
    # Set the stacklevel to 2 to print the original caller's line info
    logger.warning(msg, stacklevel=2)


class _VllmLogger(Logger):
    """
    Note:
        This class is just to provide type information.
        We actually patch the methods directly on the :class:`logging.Logger`
        instance to avoid conflicting with other libraries such as
        `intel_extension_for_pytorch.utils._logger`.
    """

    def info_once(self, msg: str) -> None:
        """
        As :meth:`info`, but subsequent calls with the same message
        are silently dropped.
        """
        _print_info_once(self, msg)

    def warning_once(self, msg: str) -> None:
        """
        As :meth:`warning`, but subsequent calls with the same message
        are silently dropped.
        """
        _print_warning_once(self, msg)


def _configure_vllm_root_logger() -> None:
    logging_config = dict[str, Any]()

    if not VLLM_CONFIGURE_LOGGING and VLLM_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "VLLM_CONFIGURE_LOGGING evaluated to false, but "
            "VLLM_LOGGING_CONFIG_PATH was given. VLLM_LOGGING_CONFIG_PATH "
            "implies VLLM_CONFIGURE_LOGGING. Please enable "
            "VLLM_CONFIGURE_LOGGING or unset VLLM_LOGGING_CONFIG_PATH.")

    if VLLM_CONFIGURE_LOGGING:
        logging_config = DEFAULT_LOGGING_CONFIG

    if VLLM_LOGGING_CONFIG_PATH:
        if not path.exists(VLLM_LOGGING_CONFIG_PATH):
            raise RuntimeError(
                "Could not load logging config. File does not exist: %s",
                VLLM_LOGGING_CONFIG_PATH)
        with open(VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())

        if not isinstance(custom_config, dict):
            raise ValueError("Invalid logging config. Expected Dict, got %s.",
                             type(custom_config).__name__)
        logging_config = custom_config

    if VLLM_LOG_FILTER_PATTERNS:
        try:
            log_filter_patterns = json.loads(VLLM_LOG_FILTER_PATTERNS)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON format in VLLM_LOG_FILTER_PATTERNS. "
                "Ensure it's a valid JSON array of strings, e.g., "
                "'[\"pattern1\", \"pattern2\"]'.") from exc
    else:
        log_filter_patterns = []

    DEFAULT_LOGGING_CONFIG["filters"]["vllm_redact"][
        "patterns"] = log_filter_patterns

    for formatter in logging_config.get("formatters", {}).values():
        # This provides backwards compatibility after #10134.
        if formatter.get("class") == "vllm.logging.NewLineFormatter":
            formatter["class"] = "vllm.logging_utils.NewLineFormatter"

    if logging_config:
        dictConfig(logging_config)


def init_logger(name: str) -> _VllmLogger:
    """The main purpose of this function is to ensure that loggers are
    retrieved in such a way that we can be sure the root vllm logger has
    already been configured."""

    logger = logging.getLogger(name)

    methods_to_patch = {
        "info_once": _print_info_once,
        "warning_once": _print_warning_once,
    }

    for method_name, method in methods_to_patch.items():
        setattr(logger, method_name, MethodType(method, logger))

    return cast(_VllmLogger, logger)


# The root logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_configure_vllm_root_logger()

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
            last_frame = frame.f_back
            if last_frame is not None:
                last_filename = last_frame.f_code.co_filename
                last_lineno = last_frame.f_lineno
                last_func_name = last_frame.f_code.co_name
            else:
                # initial frame
                last_filename = ""
                last_lineno = 0
                last_func_name = ""
            with open(log_path, 'a') as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                if event == 'call':
                    f.write(f"{ts} Call to"
                            f" {func_name} in {filename}:{lineno}"
                            f" from {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
                else:
                    f.write(f"{ts} Return from"
                            f" {func_name} in {filename}:{lineno}"
                            f" to {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
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
    logger.info("Trace frame log is saved to %s", log_file_path)
    if root_dir is None:
        # by default, this is the vllm root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))
