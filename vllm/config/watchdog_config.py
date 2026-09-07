# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import Field

from vllm.config.utils import config

# These values must be kept in sync with the defaults in
# vllm/utils/watch_dog.py (WatchDog).
_DEFAULT_TIMEOUT = 300
_DEFAULT_CHECK_INTERVAL = 10


@config
class WatchdogConfig:
    """Configuration for the stack-dump watchdog.

    The watchdog runs a background thread that periodically checks whether the
    main thread has fed it in time. If not, it dumps the stack traces of all
    threads to a dump file for debugging.
    """

    timeout: int = Field(default=_DEFAULT_TIMEOUT, gt=0)
    """Timeout in seconds after which the watchdog considers the process to be
    stuck and dumps stack traces. Must be greater than 0."""

    check_interval: int = Field(default=_DEFAULT_CHECK_INTERVAL, gt=0)
    """Interval in seconds at which the watchdog checks for feed timeouts."""

    dump_dir: str = ""
    """Directory to save stack dump files. Empty string disables the watchdog
    feature."""
