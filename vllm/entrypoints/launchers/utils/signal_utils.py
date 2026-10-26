# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import signal
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

timeout: float | None = None
shutdown_by: float | None = None
shutdown_requested = False


def to_timeout(deadline: float | None) -> float | None:
    return deadline if deadline is None else max(deadline - time.monotonic(), 0.0)


def signal_handler(signum, frame):
    global shutdown_requested
    logger.debug("Received %d signal.", signum)
    if not shutdown_requested:
        shutdown_requested = True
        raise SystemExit


def set_signal_handler() -> None:
    global shutdown_requested, shutdown_by, timeout
    shutdown_requested = False
    shutdown_by = None
    timeout = None

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def set_timeout(shutdown_timeout: float | None) -> None:
    if shutdown_timeout is None:
        return

    global timeout, shutdown_by
    if shutdown_requested:
        shutdown_by = time.monotonic() + shutdown_timeout
        logger.info("Waiting up to %d seconds for processes to exit", timeout)
