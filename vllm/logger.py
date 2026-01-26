# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logging configuration for vLLM."""

import datetime
import json
import logging
import os
import sys
from collections.abc import Generator, Hashable
from contextlib import contextmanager
from functools import lru_cache, partial
from logging import Logger
from logging.config import dictConfig
from os import path
from types import MethodType
from typing import Any, Literal, cast

import vllm.envs as envs
from vllm.logging_utils import ColoredFormatter, NewLineFormatter

_FORMAT = (
    f"{envs.VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "
    "[%(fileinfo)s:%(lineno)d] %(message)s"
)
_DATE_FORMAT = "%m-%d %H:%M:%S"


def _use_color() -> bool:
    if envs.NO_COLOR or envs.VLLM_LOGGING_COLOR == "0":
        return False
    if envs.VLLM_LOGGING_COLOR == "1":
        return True
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":  # stdout
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    elif envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":  # stderr
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    return False


DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "vllm": {
            "class": "vllm.logging_utils.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
        "vllm_color": {
            "class": "vllm.logging_utils.ColoredFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "vllm": {
            "class": "logging.StreamHandler",
            # Choose formatter based on color setting.
            "formatter": "vllm_color" if _use_color() else "vllm",
            "level": envs.VLLM_LOGGING_LEVEL,
            "stream": envs.VLLM_LOGGING_STREAM,
        },
    },
    "loggers": {
        "vllm": {
            "handlers": ["vllm"],
            "level": envs.VLLM_LOGGING_LEVEL,
            "propagate": False,
        },
    },
    "version": 1,
    "disable_existing_loggers": False,
}


@lru_cache
def _print_debug_once(logger: Logger, msg: str, *args: Hashable) -> None:
    # Set the stacklevel to 3 to print the original caller's line info
    logger.debug(msg, *args, stacklevel=3)


@lru_cache
def _print_info_once(logger: Logger, msg: str, *args: Hashable) -> None:
    # Set the stacklevel to 3 to print the original caller's line info
    logger.info(msg, *args, stacklevel=3)


@lru_cache
def _print_warning_once(logger: Logger, msg: str, *args: Hashable) -> None:
    # Set the stacklevel to 3 to print the original caller's line info
    logger.warning(msg, *args, stacklevel=3)


LogScope = Literal["process", "global", "local"]


def _should_log_with_scope(scope: LogScope) -> bool:
    """Decide whether to log based on scope"""
    if scope == "global":
        from vllm.distributed.parallel_state import is_global_first_rank

        return is_global_first_rank()
    if scope == "local":
        from vllm.distributed.parallel_state import is_local_first_rank

        return is_local_first_rank()
    # default "process" scope: always log
    return True


class _VllmLogger(Logger):
    """
    Note:
        This class is just to provide type information.
        We actually patch the methods directly on the [`logging.Logger`][]
        instance to avoid conflicting with other libraries such as
        `intel_extension_for_pytorch.utils._logger`.
    """

    def debug_once(
        self, msg: str, *args: Hashable, scope: LogScope = "process"
    ) -> None:
        """
        As [`debug`][logging.Logger.debug], but subsequent calls with
        the same message are silently dropped.
        """
        if not _should_log_with_scope(scope):
            return
        _print_debug_once(self, msg, *args)

    def info_once(self, msg: str, *args: Hashable, scope: LogScope = "process") -> None:
        """
        As [`info`][logging.Logger.info], but subsequent calls with
        the same message are silently dropped.
        """
        if not _should_log_with_scope(scope):
            return
        _print_info_once(self, msg, *args)

    def warning_once(
        self, msg: str, *args: Hashable, scope: LogScope = "process"
    ) -> None:
        """
        As [`warning`][logging.Logger.warning], but subsequent calls with
        the same message are silently dropped.
        """
        if not _should_log_with_scope(scope):
            return
        _print_warning_once(self, msg, *args)


# Pre-defined methods mapping to avoid repeated dictionary creation
_METHODS_TO_PATCH = {
    "debug_once": _VllmLogger.debug_once,
    "info_once": _VllmLogger.info_once,
    "warning_once": _VllmLogger.warning_once,
}


def _configure_vllm_root_logger() -> None:
    logging_config = dict[str, Any]()

    if not envs.VLLM_CONFIGURE_LOGGING and envs.VLLM_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "VLLM_CONFIGURE_LOGGING evaluated to false, but "
            "VLLM_LOGGING_CONFIG_PATH was given. VLLM_LOGGING_CONFIG_PATH "
            "implies VLLM_CONFIGURE_LOGGING. Please enable "
            "VLLM_CONFIGURE_LOGGING or unset VLLM_LOGGING_CONFIG_PATH."
        )

    if envs.VLLM_CONFIGURE_LOGGING:
        logging_config = DEFAULT_LOGGING_CONFIG

        vllm_handler = logging_config["handlers"]["vllm"]
        # Refresh these values in case env vars have changed.
        vllm_handler["level"] = envs.VLLM_LOGGING_LEVEL
        vllm_handler["stream"] = envs.VLLM_LOGGING_STREAM
        vllm_handler["formatter"] = "vllm_color" if _use_color() else "vllm"

        vllm_loggers = logging_config["loggers"]["vllm"]
        vllm_loggers["level"] = envs.VLLM_LOGGING_LEVEL

    if envs.VLLM_LOGGING_CONFIG_PATH:
        if not path.exists(envs.VLLM_LOGGING_CONFIG_PATH):
            raise RuntimeError(
                "Could not load logging config. File does not exist: %s",
                envs.VLLM_LOGGING_CONFIG_PATH,
            )
        with open(envs.VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())

        if not isinstance(custom_config, dict):
            raise ValueError(
                "Invalid logging config. Expected dict, got %s.",
                type(custom_config).__name__,
            )
        logging_config = custom_config

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

    for method_name, method in _METHODS_TO_PATCH.items():
        setattr(logger, method_name, MethodType(method, logger))

    return cast(_VllmLogger, logger)


@contextmanager
def suppress_logging(level: int = logging.INFO) -> Generator[None, Any, None]:
    current_level = logging.root.manager.disable
    logging.disable(level)
    yield
    logging.disable(current_level)


def current_formatter_type(lgr: Logger) -> Literal["color", "newline", None]:
    while lgr is not None:
        if lgr.handlers and len(lgr.handlers) == 1 and lgr.handlers[0].name == "vllm":
            formatter = lgr.handlers[0].formatter
            if isinstance(formatter, ColoredFormatter):
                return "color"
            if isinstance(formatter, NewLineFormatter):
                return "newline"
        lgr = lgr.parent
    return None


# The root logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_configure_vllm_root_logger()

# Transformers uses httpx to access the Hugging Face Hub. httpx is quite verbose,
# so we set its logging level to WARNING when vLLM's logging level is INFO.
if envs.VLLM_LOGGING_LEVEL == "INFO":
    logging.getLogger("httpx").setLevel(logging.WARNING)

logger = init_logger(__name__)


def _trace_calls(log_path, root_dir, frame, event, arg=None):
    if event in ["call", "return"]:
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
            with open(log_path, "a") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                if event == "call":
                    f.write(
                        f"{ts} Call to"
                        f" {func_name} in {filename}:{lineno}"
                        f" from {last_func_name} in {last_filename}:"
                        f"{last_lineno}\n"
                    )
                else:
                    f.write(
                        f"{ts} Return from"
                        f" {func_name} in {filename}:{lineno}"
                        f" to {last_func_name} in {last_filename}:"
                        f"{last_lineno}\n"
                    )
        except NameError:
            # modules are deleted during shutdown
            pass
    return partial(_trace_calls, log_path, root_dir)


def enable_trace_function_call(log_file_path: str, root_dir: str | None = None):
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
        "is suggested to be used for debugging hang or crashes only."
    )
    logger.info("Trace frame log is saved to %s", log_file_path)
    if root_dir is None:
        # by default, this is the vllm root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))
