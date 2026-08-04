# SPDX-License-Identifier: Apache-2.0

"""L1 eviction-loop metrics subscriber — OTel instruments for L1EvictionController."""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class L1EvictionLoopSubscriber(EventSubscriber):
    """Maintains OTel counters for the L1 eviction loop.

    Metrics:
    - ``lmcache_mp.l1_eviction_loop_ticks`` — every loop iteration.
    - ``lmcache_mp.l1_eviction_loop_triggered`` — iterations where
      ``usage >= watermark`` and the eviction policy ran.

    The two counters distinguish "loop is alive" from "eviction fired",
    which matters for short benchmarks that complete faster than the 1Hz
    polling rate.  Current L1 fullness is exposed separately as the
    ``lmcache_mp.l1_memory_usage_bytes`` and ``lmcache_mp.l1_usage_ratio``
    observable gauges registered in :class:`L1Manager`.
    """

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.l1")
        self._ticks = meter.create_counter(
            "lmcache_mp.l1_eviction_loop_ticks",
            description="L1 eviction-loop iterations (every cycle)",
        )
        self._triggered = meter.create_counter(
            "lmcache_mp.l1_eviction_loop_triggered",
            description="L1 eviction-loop iterations where the policy ran",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {EventType.L1_EVICTION_LOOP_TICK: self._on_tick}

    def _on_tick(self, event: Event) -> None:
        self._ticks.add(1)
        if event.metadata.get("triggered", False):
            self._triggered.add(1)
