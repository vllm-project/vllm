# SPDX-License-Identifier: Apache-2.0

"""L1 chunk lifecycle subscriber — OTel histograms for L1 chunk lifecycle.

Separated from L1MetricsSubscriber (counters) so that users can enable/disable
lifecycle tracking independently.  The shadow map and sampling overhead are
non-negligible, so this subscriber should only be registered when lifecycle
metrics are needed.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any
import time

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


@dataclass
class _L1ChunkState:
    """Per-chunk lifecycle state in the shadow map."""

    alloc_time: float
    last_access_time: float


class L1LifecycleSubscriber(EventSubscriber):
    """Tracks L1 chunk lifecycle via shadow monitoring.

    Histograms (chunk lifecycle):
    - ``lmcache_mp.l1_chunk_lifetime`` — allocation to eviction
    - ``lmcache_mp.l1_chunk_idle_before_evict`` — last access to eviction
    - ``lmcache_mp.l1_chunk_reuse_gap`` — gap between consecutive touches
    - ``lmcache_mp.l1_chunk_evict_reuse_gap`` — eviction to
      next reuse (capped at ``max_evict_reuse_wait``)

    Parameters:
        sample_rate: Fraction of chunks to track (0, 1.0].  Default 0.01 (1%).
        max_evict_reuse_wait: Maximum seconds to track an evicted chunk
            waiting for reuse.  Default 300 s (5 min).
    """

    def __init__(
        self,
        sample_rate: float = 0.01,
        max_evict_reuse_wait: float = 300.0,
    ) -> None:
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self._sample_rate = sample_rate
        self._max_evict_reuse_wait = max_evict_reuse_wait
        # Deterministic sampling via hash: hash(key) % _SAMPLE_PRIME < threshold.
        # O(1) memory, no set growth, same key always gets the same decision.
        self._sample_prime = 1_000_003
        self._sample_threshold = int(sample_rate * self._sample_prime)
        meter = metrics.get_meter("lmcache.l1")
        self._lifetime_hist = meter.create_histogram(
            "lmcache_mp.l1_chunk_lifetime",
            description=(
                "Histogram of L1 chunk lifetime from allocation to eviction (seconds)."
            ),
            unit="s",
        )
        self._idle_hist = meter.create_histogram(
            "lmcache_mp.l1_chunk_idle_before_evict",
            description=("Histogram of idle time before L1 chunk eviction (seconds)."),
            unit="s",
        )
        self._reuse_gap_hist = meter.create_histogram(
            "lmcache_mp.l1_chunk_reuse_gap",
            description=(
                "Histogram of time gaps between consecutive "
                "touches (write or read) of the same L1 chunk (seconds)."
            ),
            unit="s",
        )
        self._evict_reuse_gap_hist = meter.create_histogram(
            "lmcache_mp.l1_chunk_evict_reuse_gap",
            description=(
                "Histogram of time from L1 chunk eviction to "
                "next reuse.  Capped at max_evict_reuse_wait."
            ),
            unit="s",
        )

        # Shadow map: key -> chunk lifecycle state (live chunks).
        self._shadow: dict[Any, _L1ChunkState] = {}
        # Evicted map: key -> eviction timestamp (waiting for reuse).
        self._evicted_at: dict[Any, float] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L1_READ_FINISHED: self._on_read_finished,
            EventType.L1_WRITE_FINISHED: self._on_write_finished,
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED: self._on_write_finished,
            EventType.L1_KEYS_EVICTED: self._on_evicted,
        }

    def _on_read_finished(self, event: Event) -> None:
        now = event.timestamp or time.time()
        for key in event.metadata["keys"]:
            state = self._shadow.get(key)
            if state is not None:
                self._reuse_gap_hist.record(now - state.last_access_time)
                state.last_access_time = now

    def _on_write_finished(self, event: Event) -> None:
        now = event.timestamp or time.time()
        for key in event.metadata["keys"]:
            # Check if this is a reuse of an evicted chunk.
            evict_time = self._evicted_at.pop(key, None)
            if evict_time is not None:
                gap = min(now - evict_time, self._max_evict_reuse_wait)
                self._evict_reuse_gap_hist.record(gap)

            state = self._shadow.get(key)
            if state is not None:
                # Re-write of existing chunk counts as a touch.
                self._reuse_gap_hist.record(now - state.last_access_time)
                self._shadow[key] = _L1ChunkState(
                    alloc_time=now,
                    last_access_time=now,
                )
            else:
                # First time seeing this key — deterministic sample check.
                if not self._should_sample(key):
                    continue
                self._shadow[key] = _L1ChunkState(
                    alloc_time=now,
                    last_access_time=now,
                )
        self._sweep_stale_evictions(now)

    def _on_evicted(self, event: Event) -> None:
        now = event.timestamp or time.time()
        for key in event.metadata["keys"]:
            state = self._shadow.pop(key, None)
            if state is not None:
                self._lifetime_hist.record(now - state.alloc_time)
                self._idle_hist.record(now - state.last_access_time)
                # Start tracking eviction-to-reuse gap (only for sampled).
                self._evicted_at[key] = now
        self._sweep_stale_evictions(now)

    def _should_sample(self, key: object) -> bool:
        return hash(key) % self._sample_prime < self._sample_threshold

    def _sweep_stale_evictions(self, now: float) -> None:
        """Report T and discard evicted entries older than max_evict_reuse_wait."""
        stale = [
            key
            for key, evict_time in self._evicted_at.items()
            if now - evict_time >= self._max_evict_reuse_wait
        ]
        for key in stale:
            self._evicted_at.pop(key, None)
            self._evict_reuse_gap_hist.record(self._max_evict_reuse_wait)
