import logging
import os
import sys
from typing import Optional


_LEVEL_NAMES = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _parse_level(value: Optional[str]) -> int:
    if not value:
        return logging.WARNING
    v = value.strip().upper()
    if v.isdigit():
        return int(v)
    return _LEVEL_NAMES.get(v, logging.WARNING)


def get_weave_logger(name: str) -> logging.Logger:
    """Create/get a dedicated Weave logger.

    - Controlled by env var WEAVE_LOG_LEVEL (default: WARNING)
    - Does not propagate to the root logger to avoid mixing with vLLM logging
    - Uses its own StreamHandler (stderr) and formatter
    """

    logger_name = f"weave.{name}" if not name.startswith("weave.") else name
    logger = logging.getLogger(logger_name)

    if getattr(logger, "_weave_configured", False):
        return logger

    logger.propagate = False

    level = _parse_level(os.getenv("WEAVE_LOG_LEVEL"))
    logger.setLevel(level)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.handlers = [handler]
    logger._weave_configured = True  # type: ignore[attr-defined]
    return logger
