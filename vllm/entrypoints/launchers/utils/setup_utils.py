# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import signal
import time

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

timeout = shutdown_by = None
shutdown_requested = False


def to_timeout(deadline: float | None) -> float | None:
    return deadline if deadline is None else max(deadline - time.monotonic(), 0.0)


def signal_handler(signum, frame):
    """Catch SIGTERM and SIGINT to allow graceful shutdown."""
    global shutdown_requested
    logger.debug("Received %d signal.", signum)
    if not shutdown_requested:
        shutdown_requested = True
        raise SystemExit


def set_signal_handler():
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def set_timeout(vllm_config: VllmConfig):
    global timeout, shutdown_by

    if shutdown_requested:
        timeout = vllm_config.shutdown_timeout
        shutdown_by = time.monotonic() + timeout
        logger.info("Waiting up to %d seconds for processes to exit", timeout)
