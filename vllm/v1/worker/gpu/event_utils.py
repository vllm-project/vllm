# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Protocol

import vllm.envs as envs

# Keep ready-event latency low, then cap driver queries at 100 per second.
_INITIAL_POLL_INTERVAL_S = 0.0001
_MAX_POLL_INTERVAL_S = 0.01


class _GPUEvent(Protocol):
    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


def wait_for_gpu_event(event: _GPUEvent, operation: str) -> None:
    """Wait for a GPU event without letting an engine iteration hang forever."""
    timeout_s = envs.VLLM_ENGINE_ITERATION_TIMEOUT_S
    if timeout_s <= 0:
        event.synchronize()
        return

    deadline = time.monotonic() + timeout_s
    poll_interval_s = _INITIAL_POLL_INTERVAL_S
    while not event.query():
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError(
                f"Timed out after {timeout_s}s waiting for {operation}. "
                "The GPU may be stuck in a kernel. To adjust or disable this "
                "timeout, set VLLM_ENGINE_ITERATION_TIMEOUT_S to a positive "
                "number of seconds or 0, respectively."
            )
        time.sleep(min(poll_interval_s, remaining_s))
        poll_interval_s = min(poll_interval_s * 2, _MAX_POLL_INTERVAL_S)
