# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm.logger import init_logger

logger = init_logger(__name__)


class CircuitBreaker:
    """Open means callers skip requests; closed means requests are allowed.

    Consecutive failures reaching the threshold open it for the cooldown.
    Once it expires, a failed retry reopens it; success resets the failure count.
    """

    def __init__(self, cooldown_s: float, failure_threshold: int) -> None:
        assert cooldown_s > 0, "Circuit breaker cooldown must be positive"
        assert failure_threshold > 0, (
            "Circuit breaker failure threshold must be positive"
        )
        self.cooldown_s = cooldown_s
        self.failure_threshold = failure_threshold
        self._failures = 0
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        return time.monotonic() < self._open_until

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._open_until = time.monotonic() + self.cooldown_s
            logger.warning(
                "Mooncake lookup circuit breaker opened after %d consecutive "
                "failures; skipping lookups for %s seconds",
                self._failures,
                self.cooldown_s,
            )

    def record_success(self) -> None:
        self._failures = 0
        self._open_until = 0.0
