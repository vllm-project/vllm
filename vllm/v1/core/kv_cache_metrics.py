# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache metrics tracking."""

import random
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

from vllm.v1.metrics.stats import BlockResidencyEvent


class BlockMetricsState:
    """Tracks lifecycle metrics for a single KV cache block."""

    __slots__ = (
        "birth_time_ns",
        "last_access_ns",
        "access_history",
        "max_request_end_ns",
    )

    def __init__(self):
        now_ns = time.monotonic_ns()
        self.birth_time_ns = now_ns
        self.last_access_ns = now_ns
        self.access_history: deque[int] = deque(maxlen=4)
        self.max_request_end_ns = 0

    def record_access(self) -> None:
        now_ns = time.monotonic_ns()
        self.last_access_ns = now_ns
        self.access_history.append(now_ns)

    def get_lifetime_seconds(self, now_ns: int | None = None) -> float:
        if now_ns is None:
            now_ns = time.monotonic_ns()
        return (now_ns - self.birth_time_ns) / 1e9

    def get_idle_time_seconds(self, now_ns: int | None = None) -> float:
        if now_ns is None:
            now_ns = time.monotonic_ns()
        return (now_ns - self.last_access_ns) / 1e9

    def get_reuse_gaps_seconds(self) -> list[float]:
        if len(self.access_history) < 2:
            return []
        history = list(self.access_history)
        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]


class KVCacheMetricsCollector:
    """Collects KV cache residency metrics with sampling.

    Thread-safe via self._lock.
    """

    def __init__(self, enabled: bool = False, sample_rate: float = 0.01):
        from vllm.logger import init_logger

        logger = init_logger(__name__)

        self.enabled = enabled
        self.sample_rate = sample_rate

        # Validate sampling rate
        if sample_rate <= 0:
            self.enabled = False
            self.sample_rate = 0.0
            if sample_rate < 0:
                logger.warning("kv_cache_metrics_sample %f < 0, disabling", sample_rate)
        elif sample_rate > 1.0:
            self.sample_rate = 1.0
            logger.warning(
                "kv_cache_metrics_sample %f > 1.0, clamping to 1.0", sample_rate
            )

        self._lock = threading.Lock()
        self.block_metrics: dict[int, BlockMetricsState] = {}

        self._eviction_events: list[BlockResidencyEvent] = []

    def should_sample_block(self) -> bool:
        if not self.enabled or self.sample_rate <= 0:
            return False
        return random.random() < self.sample_rate

    def on_block_allocated(self, block: "KVCacheBlock") -> None:
        if not self.enabled:
            return
        if self.should_sample_block():
            with self._lock:
                self.block_metrics[block.block_id] = BlockMetricsState()

    def on_block_accessed(self, block: "KVCacheBlock") -> None:
        if not self.enabled:
            return
        with self._lock:
            metrics = self.block_metrics.get(block.block_id)
            if metrics:
                metrics.record_access()

    def on_block_evicted(self, block: "KVCacheBlock") -> None:
        if not self.enabled:
            return

        with self._lock:
            metrics = self.block_metrics.pop(block.block_id, None)
            if not metrics:
                return

            now_ns = time.monotonic_ns()
            lifetime = metrics.get_lifetime_seconds(now_ns)
            idle_time = metrics.get_idle_time_seconds(now_ns)
            reuse_gaps = tuple(metrics.get_reuse_gaps_seconds())
            prefix_residency: float | None = None
            if metrics.max_request_end_ns > 0:
                residency = (now_ns - metrics.max_request_end_ns) / 1e9
                if residency >= 0:
                    prefix_residency = residency

            self._eviction_events.append(
                BlockResidencyEvent(
                    lifetime_seconds=lifetime,
                    idle_seconds=idle_time,
                    reuse_gaps_seconds=reuse_gaps,
                    prefix_residency_seconds=prefix_residency,
                )
            )

    def on_request_prefill_complete(
        self, request_id: str, prefix_block_ids: set[int]
    ) -> None:
        """Track prefix blocks. Handles multiple requests sharing same blocks."""
        if not self.enabled:
            return

        now_ns = time.monotonic_ns()
        with self._lock:
            for block_id in prefix_block_ids:
                metrics = self.block_metrics.get(block_id)
                if metrics:
                    metrics.max_request_end_ns = max(metrics.max_request_end_ns, now_ns)

    def reset(self) -> None:
        """Clear all state on cache reset."""
        with self._lock:
            self.block_metrics.clear()
            self._eviction_events.clear()

    def drain_events(self) -> list[BlockResidencyEvent]:
        if not self.enabled:
            return []
        with self._lock:
            events = self._eviction_events
            self._eviction_events = []
        return events
