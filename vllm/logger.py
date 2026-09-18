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
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache, partial
from logging import Logger
from logging.config import dictConfig
from os import path
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal, cast

import vllm.envs as envs
from vllm.logging_utils import ColoredFormatter, NewLineFormatter

if TYPE_CHECKING:
    from vllm.config.logging import LoggingConfig

_FORMAT = (
    f"{envs.VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "
    "[%(fileinfo)s:%(lineno)d] %(message)s"
)
_DATE_FORMAT = "%m-%d %H:%M:%S"


def _use_color() -> bool:
    if envs.NO_COLOR or envs.VLLM_LOGGING_COLOR == "0":
        return False
    if envs.VLLM_LOGGING_COLOR == "1" or envs.FORCE_COLOR:
        return True
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":  # stdout
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    elif envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":  # stderr
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    return False


DEFAULT_LOGGING_CONFIG: dict[str, dict[str, Any] | Any] = {
    "formatters": {
        # The "()" factory form forwards custom kwargs such as log_level.
        "vllm": {
            "()": "vllm.logging_utils.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
            "log_level": envs.VLLM_LOGGING_LEVEL,
        },
        "vllm_color": {
            "()": "vllm.logging_utils.ColoredFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
            "log_level": envs.VLLM_LOGGING_LEVEL,
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

_last_configured_logging_config: "LoggingConfig | None" = None


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


@lru_cache
def _print_error_once(logger: Logger, msg: str, *args: Hashable) -> None:
    # Set the stacklevel to 3 to print the original caller's line info
    logger.error(msg, *args, stacklevel=3)


LogScope = Literal["process", "global", "local"]


def _should_log_with_scope(scope: LogScope) -> bool:
    """Decide whether to log based on scope."""
    if scope == "global":
        from vllm.distributed.parallel_state import is_global_first_rank

        return is_global_first_rank()
    if scope == "local":
        from vllm.distributed.parallel_state import is_local_first_rank

        return is_local_first_rank()
    return True


class _VllmLogger(Logger):
    """Note:
    This class is just to provide type information.
    We actually patch the methods directly on the [`logging.Logger`][]
    instance to avoid conflicting with other libraries such as
    `intel_extension_for_pytorch.utils._logger`.

    """

    def debug_once(self, msg: str, *args: Hashable, scope: LogScope = "local") -> None:
        """As [`debug`][logging.Logger.debug], but subsequent calls with
        the same message are silently dropped.
        """
        if not _should_log_with_scope(scope):
            return
        _print_debug_once(self, msg, *args)

    def info_once(self, msg: str, *args: Hashable, scope: LogScope = "local") -> None:
        """As [`info`][logging.Logger.info], but subsequent calls with
        the same message are silently dropped.
        """
        if not _should_log_with_scope(scope):
            return
        _print_info_once(self, msg, *args)

    def warning_once(
        self, msg: str, *args: Hashable, scope: LogScope = "local"
    ) -> None:
        """As [`warning`][logging.Logger.warning], but subsequent calls with
        the same message are silently dropped.
        """
        if not _should_log_with_scope(scope):
            return
        _print_warning_once(self, msg, *args)

    def error_once(self, msg: str, *args: Hashable, scope: LogScope = "local") -> None:
        """As [`error`][logging.Logger.error], but subsequent calls with
        the same message are silently dropped.
        """
        if not _should_log_with_scope(scope):
            return
        _print_error_once(self, msg, *args)


# Pre-defined methods mapping to avoid repeated dictionary creation
_METHODS_TO_PATCH = {
    "debug_once": _VllmLogger.debug_once,
    "info_once": _VllmLogger.info_once,
    "warning_once": _VllmLogger.warning_once,
    "error_once": _VllmLogger.error_once,
}


def _configure_vllm_root_logger(config: "LoggingConfig | None" = None) -> None:
    """Configure logging from explicit config or bootstrap environment values."""
    logging_config: dict[str, dict[str, Any] | Any] = {}
    if config is None:
        configure_logging = envs.VLLM_CONFIGURE_LOGGING
        log_level = envs.VLLM_LOGGING_LEVEL
        log_config_file = envs.VLLM_LOGGING_CONFIG_PATH
    else:
        configure_logging = config.configure_logging
        log_level = config.log_level
        log_config_file = config.pylogging_config_file

    if not configure_logging and log_config_file:
        raise RuntimeError(
            "Logging configuration is disabled, but a Python logging config "
            "file was given. pylogging_config_file requires "
            "configure_logging to be enabled."
        )

    if configure_logging:
        logging_config = deepcopy(DEFAULT_LOGGING_CONFIG)

        vllm_handler = logging_config["handlers"]["vllm"]
        # Refresh these values in case env vars have changed.
        vllm_handler["level"] = log_level
        vllm_handler["stream"] = envs.VLLM_LOGGING_STREAM
        vllm_handler["formatter"] = "vllm_color" if _use_color() else "vllm"

        vllm_loggers = logging_config["loggers"]["vllm"]
        vllm_loggers["level"] = log_level
        for formatter in logging_config["formatters"].values():
            formatter["log_level"] = log_level

    if log_config_file:
        if not path.exists(log_config_file):
            raise RuntimeError(
                "Could not load logging config. File does not exist: %s",
                log_config_file,
            )
        with open(log_config_file, encoding="utf-8") as file:
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

    # Transformers uses httpx to access the Hugging Face Hub. httpx is quite verbose,
    # so we set its logging level to WARNING when vLLM's logging level is INFO.
    # httpx2 is the successor huggingface_hub switches to in its 2.x releases.
    if log_level == "INFO":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpx2").setLevel(logging.WARNING)


def configure_logging(config: "LoggingConfig") -> None:
    """Apply a logging configuration in the current process."""
    _configure_vllm_root_logger(config)
    global _last_configured_logging_config
    _last_configured_logging_config = config
    _log_platform_warnings(config)


def _log_platform_warnings(config: "LoggingConfig") -> None:
    """Emit platform diagnostics only after an enabled config is active."""
    if not config.configure_logging:
        return

    # Import lazily because platform modules use init_logger during import.
    from vllm.platforms import current_platform

    current_platform.log_warnings()


def configure_logging_if_needed(config: "LoggingConfig") -> None:
    """Apply a logging configuration unless it is already active in this process."""
    if config != _last_configured_logging_config:
        configure_logging(config)


def configure_logging_from_args(args: Any) -> "LoggingConfig":
    """Apply parsed logging arguments and retain them for child processes."""
    from vllm.config.logging import LoggingConfig

    config = getattr(args, "logging_config", None) or LoggingConfig()
    if hasattr(args, "log_level"):
        config = replace(config, log_level=args.log_level)
    if hasattr(args, "log_config_file"):
        config = replace(config, pylogging_config_file=args.log_config_file)

    configure_logging_if_needed(config)
    args.log_config_file = config.pylogging_config_file
    args.logging_config = config
    return config


def init_logger(name: str) -> _VllmLogger:
    """Retrieve a logger and add vLLM's convenience logging methods."""
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


def current_formatter_type(logger: Logger) -> Literal["color", "newline", None]:
    lgr: Logger | None = logger
    while lgr is not None:
        if lgr.handlers and len(lgr.handlers) == 1 and lgr.handlers[0].name == "vllm":
            formatter = lgr.handlers[0].formatter
            if isinstance(formatter, ColoredFormatter):
                return "color"
            if isinstance(formatter, NewLineFormatter):
                return "newline"
        lgr = lgr.parent
    return None


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
    """Enable tracing of every function call in code under `root_dir`.
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
