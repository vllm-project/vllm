# SPDX-License-Identifier: Apache-2.0

"""OTel counter for timeout errors: ``lmcache_mp.timeouts`` by ``exception_type``."""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class TimeoutMetricsSubscriber(EventSubscriber):
    """Maintains an OTel counter for timeout errors."""

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache_mp.health")
        self._timeout_counter = meter.create_counter(
            "lmcache_mp.timeouts",
            description="Timeouts raised, tagged by exception_type.",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {EventType.TIMEOUT_RAISED: self._on_timeout}

    def _on_timeout(self, event: Event) -> None:
        exception_type: str = event.metadata["exception_type"]
        self._timeout_counter.add(1, {"exception_type": exception_type})
