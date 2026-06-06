# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from collections import deque

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


class PercentileTracker:
    """Track acceptance lengths and filter by worst percentile.

    Uses a two-tier check: values strictly below the threshold are always
    captured, values exactly at the threshold are randomly captured to
    approximate the target percentile. This handles uniform distributions
    where all values cluster at the same level.

    Args:
        percentile: Capture drafts in the worst X% (0-100).
        window_size: Sliding window of recent observations.
        min_samples: Minimum observations before filtering kicks in.
            During warmup, all requests are captured.
    """

    def __init__(
        self,
        percentile: float = 10.0,
        window_size: int = 1000,
        min_samples: int = 100,
    ):
        self.percentile = percentile
        self.window_size = window_size
        self.min_samples = min_samples

        self._window: deque[float] = deque(maxlen=window_size)
        self._cached_threshold: float | None = None
        self._fraction_at_threshold: float = 0.0
        self._cache_valid = False

    def observe(self, acceptance_length: float) -> None:
        """Record an acceptance length observation."""
        self._window.append(acceptance_length)
        if len(self._window) % 10 == 0:
            self._cache_valid = False

    def should_capture(self, acceptance_length: float) -> bool:
        """Check if this acceptance length is in the worst percentile.

        Returns True if the value should be captured (i.e. it's bad enough
        to be useful for distillation training).
        """
        if len(self._window) < self.min_samples:
            return True

        if not self._cache_valid:
            self.update_threshold()

        assert self._cached_threshold is not None

        if acceptance_length < self._cached_threshold:
            return True
        if acceptance_length > self._cached_threshold:
            return False
        # At the threshold: randomly capture to hit target percentage
        return random.random() < self._fraction_at_threshold

    def observe_and_check(self, acceptance_length: float) -> bool:
        """Atomically observe and check whether to capture."""
        should = self.should_capture(acceptance_length)
        self.observe(acceptance_length)
        return should

    def update_threshold(self) -> None:
        """Recompute the cached percentile threshold and boundary fraction.

        The fraction_at_threshold handles the case where many values equal
        the threshold. For example, if percentile=5 and all values are 2.0,
        the threshold is 2.0, fraction_below is 0%, so we need to randomly
        capture 5% of the at-threshold values.
        """
        if len(self._window) == 0:
            self._cached_threshold = float("inf")
            self._fraction_at_threshold = 0.0
            return
        arr = np.array(self._window)
        self._cached_threshold = float(np.percentile(arr, self.percentile))
        fraction_below = float(np.mean(arr < self._cached_threshold))
        fraction_at = float(np.mean(arr == self._cached_threshold))
        target = self.percentile / 100.0
        if fraction_at > 0:
            self._fraction_at_threshold = max(
                0.0, min(1.0, (target - fraction_below) / fraction_at))
        else:
            self._fraction_at_threshold = 0.0
        self._cache_valid = True

    def get_stats(self) -> dict:
        """Return current tracker statistics."""
        if len(self._window) == 0:
            return {
                "num_samples": 0,
                "percentile_threshold": None,
                "mean_acceptance": None,
            }
        if not self._cache_valid:
            self.update_threshold()
        arr = np.array(self._window)
        return {
            "num_samples": len(self._window),
            "percentile_threshold": self._cached_threshold,
            "fraction_at_threshold": self._fraction_at_threshold,
            "mean_acceptance": float(np.mean(arr)),
            "min_acceptance": float(np.min(arr)),
            "max_acceptance": float(np.max(arr)),
        }
