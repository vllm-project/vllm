# SPDX-License-Identifier: Apache-2.0

"""L2 failure metrics subscriber — OTel counters for L2 prefetch failures.

This covers the health-monitoring surface for L2 prefetch (see LM-291):

- ``lmcache_mp.l2_prefetch_failure`` — count of keys that failed to load from
  L2 to L1. Tagged by ``reason``:
    * ``l1_oom``    — L1 had no room to receive the prefetched object.
    * ``not_found`` — L2 reported the key present during lookup but the load
      returned no data (adapter-level inconsistency, e.g. concurrent delete).

The ``serde_failure`` reason is intentionally omitted until the serde PR
lands; once it does, it becomes an additive third value of the same tag
with no breaking change to dashboards.

All emissions carry ``model_name`` extracted from each ``ObjectKey``.
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


class L2FailureMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for L2 prefetch failures."""

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache_mp.health")
        self._prefetch_counter = meter.create_counter(
            "lmcache_mp.l2_prefetch_failure",
            description=(
                "Count of keys that were expected in L2 but failed to load "
                "into L1. Tagged by ``reason`` = l1_oom | not_found and "
                "``model_name``."
            ),
            unit="chunks",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {EventType.L2_PREFETCH_FAILED: self._on_prefetch_failed}

    def _on_prefetch_failed(self, event: Event) -> None:
        reason: str = event.metadata["reason"]
        keys: list[ObjectKey] = event.metadata["keys"]
        for model_name, count in Counter(k.model_name for k in keys).items():
            self._prefetch_counter.add(
                count,
                {"reason": reason, "model_name": model_name},
            )
