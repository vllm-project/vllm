# SPDX-License-Identifier: Apache-2.0

"""L1 failure metrics subscriber — OTel counters for L1 allocation and read failures.

These counters cover the health-monitoring surface for L1 (see LM-291):

- ``lmcache_mp.l1_allocation_failure`` — L1 memory allocation failures (OOM)
  during reserve_write. Tagged by ``during`` (``l1_store`` vs ``l2_prefetch``)
  to distinguish user-initiated stores from prefetch-triggered allocations.
- ``lmcache_mp.l1_read_failure`` — L1 reserve_read failures. This is an
  anomaly counter, not a cache-miss counter: in MP mode ``reserve_read`` is
  only called after a successful lookup, so any failure here indicates a
  lookup/reserve race or unexpected eviction and should stay near zero in
  healthy operation.

All counters carry ``model_name`` extracted from each ``ObjectKey`` so operators
can slice by model on the Prometheus ``/metrics`` endpoint.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class L1FailureMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for L1 allocation and read failures."""

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache_mp.health")
        self._allocation_counter = meter.create_counter(
            "lmcache_mp.l1_allocation_failure",
            description=(
                "Count of L1 memory allocation failures (OOM) during "
                "reserve_write. Tagged by ``during`` = l1_store | l2_prefetch "
                "and ``model_name``."
            ),
            unit="chunks",
        )
        self._read_counter = meter.create_counter(
            "lmcache_mp.l1_read_failure",
            description=(
                "Count of L1 reserve_read failures (post-lookup anomaly). "
                "Tagged by ``during`` = l2_store | l1_retrieve, ``reason`` = "
                "not_found | write_locked, and ``model_name``. Should stay "
                "near zero in healthy operation; non-zero indicates a "
                "lookup/reserve race or unexpected eviction."
            ),
            unit="chunks",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L1_ALLOCATION_FAILED: self._on_allocation_failed,
            EventType.L1_READ_FAILED: self._on_read_failed,
        }

    def _on_allocation_failed(self, event: Event) -> None:
        during: str = event.metadata["during"]
        keys: list[ObjectKey] = event.metadata["keys"]
        for model_name, count in Counter(k.model_name for k in keys).items():
            self._allocation_counter.add(
                count,
                {"during": during, "model_name": model_name},
            )

    def _on_read_failed(self, event: Event) -> None:
        during: str = event.metadata["during"]
        reason: str = event.metadata["reason"]
        keys: list[ObjectKey] = event.metadata["keys"]
        for model_name, count in Counter(k.model_name for k in keys).items():
            self._read_counter.add(
                count,
                {
                    "during": during,
                    "reason": reason,
                    "model_name": model_name,
                },
            )
