# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache metrics tracking."""

import random
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

from vllm.v1.metrics.stats import KVCacheEvictionEvent


class BlockMetricsState:
    """Tracks lifecycle metrics for a single KV cache block."""

    def __init__(self):
        now_ns = time.monotonic_ns()
        self.birth_time_ns = now_ns
        self.last_access_ns = now_ns
        # Bounded to prevent unbounded growth if a block is accessed many times.
        self.access_history: deque[int] = deque(maxlen=4)

    def record_access(self) -> None:
        now_ns = time.monotonic_ns()
        self.last_access_ns = now_ns
        self.access_history.append(now_ns)

    def get_lifetime_seconds(self) -> float:
        now_ns = time.monotonic_ns()
        return (now_ns - self.birth_time_ns) / 1e9

    def get_idle_time_seconds(self) -> float:
        now_ns = time.monotonic_ns()
        return (now_ns - self.last_access_ns) / 1e9

    def get_reuse_gaps_seconds(self) -> list[float]:
        if len(self.access_history) < 2:
            return []
        history = list(self.access_history)
        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]


class KVCacheMetricsCollector:
    """Collects KV cache residency metrics with sampling."""

    def __init__(self, sample_rate: float = 0.01):
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self.sample_rate = sample_rate

        self.block_metrics: dict[int, BlockMetricsState] = {}

        self._eviction_events: list[KVCacheEvictionEvent] = []

    def should_sample_block(self) -> bool:
        return random.random() < self.sample_rate

    def on_block_allocated(self, block: "KVCacheBlock") -> None:
        if self.should_sample_block():
            self.block_metrics[block.block_id] = BlockMetricsState()

    def on_block_accessed(self, block: "KVCacheBlock") -> None:
        metrics = self.block_metrics.get(block.block_id)
        if metrics:
            metrics.record_access()

    def on_block_evicted(self, block: "KVCacheBlock") -> None:
        metrics = self.block_metrics.pop(block.block_id, None)
        if not metrics:
            return

        lifetime = metrics.get_lifetime_seconds()
        idle_time = metrics.get_idle_time_seconds()
        reuse_gaps = tuple(metrics.get_reuse_gaps_seconds())

        self._eviction_events.append(
            KVCacheEvictionEvent(
                lifetime_seconds=lifetime,
                idle_seconds=idle_time,
                reuse_gaps_seconds=reuse_gaps,
            )
        )

    def reset(self) -> None:
        """Clear all state on cache reset."""
        self.block_metrics.clear()
        self._eviction_events.clear()

    def drain_events(self) -> list[KVCacheEvictionEvent]:
        events = self._eviction_events
        self._eviction_events = []
        return events
