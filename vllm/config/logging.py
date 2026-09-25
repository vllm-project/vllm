# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, cast

from pydantic import Field

import vllm.envs as envs
from vllm.config.utils import config

LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]


@config
class LoggingConfig:
    """vLLM logging. Default ``dictConfig`` adds a formatted stream handler.

    Supply a JSON object with ``--logging-config`` or individual fields with
    dotted arguments such as ``--logging-config.log_level DEBUG`` and
    ``--logging-config.pylogging_config_file logging.json``. ``--log-level``
    and legacy ``--log-config-file`` override their matching fields. Set
    ``configure_logging`` to false to skip applying a ``dictConfig``.
    """

    log_level: LogLevel = Field(
        default_factory=lambda: cast(LogLevel, envs.VLLM_LOGGING_LEVEL)
    )
    """Log level used when no custom logging configuration is provided."""

    configure_logging: bool = Field(default_factory=lambda: envs.VLLM_CONFIGURE_LOGGING)
    """Whether to apply a Python logging configuration.

    With no ``pylogging_config_file``, the default config adds a formatted
    stream handler to the ``vllm`` logger. A custom Python logging configuration
    file also requires this to be enabled.
    """

    pylogging_config_file: str | None = Field(
        default_factory=lambda: envs.VLLM_LOGGING_CONFIG_PATH
    )
    """Path to a Python logging JSON file using
    [``dictConfig`` schema](https://docs.python.org/3/library/logging.config.html#configuration-file-format).

    A custom ``dictConfig`` is authoritative over ``log_level`` for logger,
    handler, and formatter settings.
    """
