# SPDX-License-Identifier: Apache-2.0
"""Logging configuration for vLLM."""

import datetime
import inspect
import json
import logging
import os
import sys
import time
import traceback
from functools import lru_cache, partial
from logging import Logger
from logging.config import dictConfig
from os import path
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, Optional, Union, cast

from loguru import logger
from loguru._logger import Logger as LoguruLogger

import vllm.envs as envs

VLLM_CONFIGURE_LOGGING = envs.VLLM_CONFIGURE_LOGGING
VLLM_LOGGING_CONFIG_PATH = envs.VLLM_LOGGING_CONFIG_PATH
VLLM_LOGGING_LEVEL = envs.VLLM_LOGGING_LEVEL
VLLM_LOGGING_PREFIX = envs.VLLM_LOGGING_PREFIX

_FORMAT = (f"{VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "
           "[%(filename)s:%(lineno)d] %(message)s")
_DATE_FORMAT = "%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "vllm": {
            "class": "vllm.logging_utils.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "vllm": {
            "class": "logging.StreamHandler",
            "formatter": "vllm",
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
    "disable_existing_loggers": False,
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


class InterceptHandler(logging.Handler):
    """A custom logging handler for intercepting logs and redirecting them.

    This handler is designed to integrate with loguru's `logger` library,
    capturing log records generated by Python's standard logging module and
    processing them using `logger`.

    Attributes:
        loglevel_mapping (dict): Mapping of logging level numbers to string
        representations.
    """

    loglevel_mapping = {
        50: "CRITICAL",
        40: "ERROR",
        30: "WARNING",
        20: "INFO",
        10: "DEBUG",
        0: "NOTSET",
    }

    def emit(self, record: logging.LogRecord) -> None:
        """Process the logging record.

        Overrides the default emit method to handle log records. It binds the
        logger to a specific name and adds contextual information.

        Args:
            record (logging.LogRecord): The log record to process.
        """
        try:
            level = logger.level(record.levelname).name
        except AttributeError:
            level = self.loglevel_mapping[record.levelno]

        # Use the improved frame detection from the newer code
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0
                         or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth,
                   exception=record.exc_info).log(level, record.getMessage())


class CustomizeLogger:
    """A class for configuring and customizing logging.

    Provides methods for setting up structured and unstructured logging,
    including log rotation, retention, and formatting.

    Methods are class methods, allowing direct access without instantiation.
    """

    @classmethod
    def make_logger(cls, config_path: Path) -> LoguruLogger:
        """Create and configure a logger based on a config file.

        Args:
            config_path: Path to the logging configuration file.

        Returns:
            A configured logger instance.
        """
        config = cls.load_logging_config(config_path)
        logging_config = config.get("logger")

        params = {
            "structured_filepath":
            logging_config.get("structured_log_file_path"),
            "unstructured_filepath":
            logging_config.get("unstructured_log_file_path"),
            "level":
            logging_config.get("level"),
            "retention":
            logging_config.get("retention"),
            "rotation":
            logging_config.get("rotation"),
            "format":
            logging_config.get("format"),
        }

        _logger = cls.customize_logging(**params)

        return _logger

    @classmethod
    def serialize(cls, record: Dict[str, Any]) -> str:
        """Serialize a log record to JSON.

        Args:
            record: The log record to serialize.

        Returns:
            The serialized JSON string.
        """

        def format_exception(ex: Exception) -> str:
            """Format exception to include the entire traceback chain."""
            lines = []
            while ex:
                lines.extend(traceback.format_exception_only(ex.__class__, ex))
                tb = ex.__traceback__
                while tb:
                    lines.extend(traceback.format_tb(tb))
                    tb = tb.tb_next
                ex = ex.__cause__ or ex.__context__

            return "".join(lines)

        exception = record["exception"]
        error = ""
        if exception:
            _, ex, _ = exception
            error = f" {ex.__class__.__name__}: {ex}\n{format_exception(ex)}"

        subset = {
            "module": record["module"],
            "pathname": record["file"].name,
            "lineno": record["line"],
            "thread": record["thread"].id,
            "extra_info": record["extra"],
            "funcName": record["function"],
            "ts": int(time.time() * 1000),
            "level": record["level"].name,
            "msg": record["message"] + error,
        }
        return json.dumps(subset)

    @classmethod
    def formatter(cls, record: Dict[str, Any]) -> str:
        """Format a log record for structured logging.

        Note: This function returns the string to be formatted, not the actual
        message to be logged.

        Args:
            record: The log record to format.

        Returns:
            The formatted log record.
        """
        record["extra"]["serialized"] = cls.serialize(record)
        return "{extra[serialized]}\n"

    @classmethod
    def customize_logging(
        cls,
        structured_filepath: Optional[Path],
        unstructured_filepath: Optional[Path],
        level: Optional[str],
        rotation: Optional[str],
        retention: Optional[str],
        format: Optional[str],
    ) -> LoguruLogger:
        """Customize logging setup based on configuration options.

        Configures logging with options for both structured and unstructured
        log files, including rotation, retention, and formatting. Allows
        enabling/disabling file logging and structured logging.

        Args:
            structured_filepath: Path to the structured log file.
            unstructured_filepath: Path to the unstructured log file.
            level: Logging level.
            rotation: Log rotation policy.
            retention: Log retention policy.
            format: Log format.

        Returns:
            Configured logger instance.
        """
        logger.remove()
        logger.add(
            sys.stdout,
            enqueue=True,
            backtrace=True,
            level=level.upper() if level else "INFO",
            format=format
            or ("<level>{level: <8}</level> "
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - "
                "<level>{message}</level>"),
        )

        if unstructured_filepath:
            # Unstructured file logging configuration
            logger.add(
                str(unstructured_filepath),
                rotation=rotation or "1 days",
                retention=retention or "1 weeks",
                enqueue=True,
                backtrace=True,
                level=(level or "INFO").upper(),
                serialize=False,
                format=format
                or ("<level>{level: <8}</level> "
                    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - "
                    "<level>{message}</level>"),
            )
        if structured_filepath:
            # Structured file logging configuration
            logger.add(
                str(structured_filepath),
                rotation=rotation or "1 days",
                retention=retention or "1 weeks",
                enqueue=True,
                backtrace=True,
                level=(level or "INFO").upper(),
                serialize=False,
                format=cls.formatter,
            )

        # Only set up a handler for the vllm logger
        vllm_logger = logging.getLogger("vllm")
        for handler in vllm_logger.handlers[:]:
            vllm_logger.removeHandler(handler)
        vllm_logger.addHandler(InterceptHandler())
        vllm_logger.setLevel(0)
        vllm_logger.propagate = False

        return logger.bind(name="vllm")

    @classmethod
    def load_logging_config(cls, config_path: Path) -> Dict[str, Any]:
        """Load logging configuration from a JSON file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Loaded configuration.
        """
        config = None
        with open(config_path) as config_file:
            config = json.load(config_file)
        return config


def _configure_vllm_root_logger() -> None:
    """Configure the root vLLM logger based on environment variables
    or JSON config."""
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
                VLLM_LOGGING_CONFIG_PATH,
            )
        with open(VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())

        if not isinstance(custom_config, dict):
            raise ValueError(
                "Invalid logging config. Expected Dict, got %s.",
                type(custom_config).__name__,
            )
        logging_config = custom_config

    for formatter in logging_config.get("formatters", {}).values():
        # This provides backwards compatibility after #10134.
        if formatter.get("class") == "vllm.logging.NewLineFormatter":
            formatter["class"] = "vllm.logging_utils.NewLineFormatter"

    if logging_config:
        dictConfig(logging_config)


# Try to find the loguru config path
dir_path = os.path.dirname(os.path.realpath(__file__))
default_config_path = f"{dir_path}/default_logging_config.json"
config_path = Path(os.getenv("VLLM_LOGGING_CONFIG_PATH", default_config_path))
_root_logger = None
_default_handler = None


def _setup_logger():
    """Configure the vllm logger with basic settings when 
    VLLM_CONFIGURE_LOGGING is False."""
    global _default_handler, _root_logger
    _root_logger.setLevel(logging.DEBUG)
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    from vllm.logging_utils import NewLineFormatter

    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message being propagated to the parent logger
    _root_logger.propagate = False


# Initialize the logger based on configuration
if VLLM_CONFIGURE_LOGGING:
    if os.path.exists(config_path):
        _root_logger = CustomizeLogger.make_logger(config_path)
    else:
        _configure_vllm_root_logger()
        _root_logger = None
else:
    _root_logger = logging.getLogger("vllm")
    _setup_logger()


def init_logger(name: str) -> Union[LoguruLogger, _VllmLogger]:
    """Initialize a logger for the given name.

    This function supports both standard Python logging and loguru logging
    based on VLLM_CONFIGURE_LOGGING setting.

    Args:
        name: The name of the logger.

    Returns:
        A logger instance (either a loguru logger or standard Logger
        with custom methods).
    """
    if VLLM_CONFIGURE_LOGGING and _root_logger and hasattr(
            _root_logger, 'bind'):
        return _root_logger.bind(name=name)

    logger = logging.getLogger(name)

    if not VLLM_CONFIGURE_LOGGING:
        logger.setLevel(os.getenv("LOG_LEVEL", "DEBUG"))

    methods_to_patch = {
        "info_once": _print_info_once,
        "warning_once": _print_warning_once,
    }

    for method_name, method in methods_to_patch.items():
        setattr(logger, method_name, MethodType(method, logger))

    return cast(_VllmLogger, logger)


module_logger = init_logger(__name__)


def _trace_calls(log_path: str,
                 root_dir: str,
                 frame: Any,
                 event: str,
                 arg: Any = None) -> Callable:
    """Trace function calls and log them to a file.

    Args:
        log_path: Path to the log file.
        root_dir: Root directory to filter traced functions.
        frame: Current frame being executed.
        event: Event type (call or return).
        arg: Additional arguments.

    Returns:
        A callable for the next trace call.
    """
    if event in ["call", "return"]:
        # Extract the filename, line number, function name, and the code object
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        if not filename.startswith(root_dir):
            # only log the functions in the vllm root_dir
            return partial(_trace_calls, log_path, root_dir)
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
                               root_dir: Optional[str] = None) -> None:
    """
    Enable tracing of every function call in code under `root_dir`.
    This is useful for debugging hangs or crashes.
    `log_file_path` is the path to the log file.
    `root_dir` is the root directory of the code to trace. If None, it is the
    vllm root directory.

    Note that this call is thread-level, any threads calling this function
    will have the trace enabled. Other threads will not be affected.
    """
    module_logger.warning(
        "VLLM_TRACE_FUNCTION is enabled. It will record every"
        " function executed by Python. This will slow down the code. It "
        "is suggested to be used for debugging hang or crashes only.")
    module_logger.info("Trace frame log is saved to %s", log_file_path)
    if root_dir is None:
        # by default, this is the vllm root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))
