# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percentile-based acceptance rate tracker for capture filtering.

Maintains a sliding window of acceptance lengths and determines
whether a given observation falls in the worst X percentile.
Only requests with poor acceptance rates are captured, focusing
data collection on the cases most useful for distillation.
"""

from collections import deque
from typing import Optional

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


class PercentileTracker:
    """Track acceptance lengths and filter by worst percentile.

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
        self._cached_threshold: Optional[float] = None
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
            # Warmup: capture everything
            return True

        if not self._cache_valid:
            self._update_threshold()

        assert self._cached_threshold is not None
        return bool(acceptance_length <= self._cached_threshold)

    def observe_and_check(self, acceptance_length: float) -> bool:
        """Atomically observe and check whether to capture."""
        should = self.should_capture(acceptance_length)
        self.observe(acceptance_length)
        return should

    def _update_threshold(self) -> None:
        """Recompute the cached percentile threshold."""
        if len(self._window) == 0:
            self._cached_threshold = float("inf")
            return
        arr = np.array(self._window)
        self._cached_threshold = float(np.percentile(arr, self.percentile))
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
            self._update_threshold()
        arr = np.array(self._window)
        return {
            "num_samples": len(self._window),
            "percentile_threshold": self._cached_threshold,
            "mean_acceptance": float(np.mean(arr)),
            "min_acceptance": float(np.min(arr)),
            "max_acceptance": float(np.max(arr)),
        }
