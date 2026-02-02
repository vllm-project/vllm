# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Access log filter for uvicorn to exclude specific endpoints from logging.

This module provides a logging filter that can be used to suppress access logs
for specific endpoints (e.g., /health, /metrics) to reduce log noise in
production environments.
"""

import logging
from urllib.parse import urlparse


class UvicornAccessLogFilter(logging.Filter):
    """
    A logging filter that excludes access logs for specified endpoint paths.

    This filter is designed to work with uvicorn's access logger. It checks
    the log record's arguments for the request path and filters out records
    matching the excluded paths.

    Uvicorn access log format:
        '%s - "%s %s HTTP/%s" %d'
        (client_addr, method, path, http_version, status_code)

    Example:
        127.0.0.1:12345 - "GET /health HTTP/1.1" 200

    Args:
        excluded_paths: A list of URL paths to exclude from logging.
                       Paths are matched exactly.
                       Example: ["/health", "/metrics"]
    """

    def __init__(self, excluded_paths: list[str] | None = None):
        super().__init__()
        self.excluded_paths = set(excluded_paths or [])

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Determine if the log record should be logged.

        Args:
            record: The log record to evaluate.

        Returns:
            True if the record should be logged, False otherwise.
        """
        if not self.excluded_paths:
            return True

        # This filter is specific to uvicorn's access logs.
        if record.name != "uvicorn.access":
            return True

        # The path is the 3rd argument in the log record's args tuple.
        # See uvicorn's access logging implementation for details.
        log_args = record.args
        if isinstance(log_args, tuple) and len(log_args) >= 3:
            path_with_query = log_args[2]
            # Get path component without query string.
            if isinstance(path_with_query, str):
                path = urlparse(path_with_query).path
                if path in self.excluded_paths:
                    return False

        return True


def create_uvicorn_log_config(
    excluded_paths: list[str] | None = None,
    log_level: str = "info",
) -> dict:
    """
    Create a uvicorn logging configuration with access log filtering.

    This function generates a logging configuration dictionary that can be
    passed to uvicorn's `log_config` parameter. It sets up the access log
    filter to exclude specified paths.

    Args:
        excluded_paths: List of URL paths to exclude from access logs.
        log_level: The log level for uvicorn loggers.

    Returns:
        A dictionary containing the logging configuration.

    Example:
        >>> config = create_uvicorn_log_config(["/health", "/metrics"])
        >>> uvicorn.run(app, log_config=config)
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "access_log_filter": {
                "()": UvicornAccessLogFilter,
                "excluded_paths": excluded_paths or [],
            },
        },
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',  # noqa: E501
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "filters": ["access_log_filter"],
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": log_level.upper(),
                "propagate": False,
            },
            "uvicorn.error": {
                "level": log_level.upper(),
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": log_level.upper(),
                "propagate": False,
            },
        },
    }
    return config
