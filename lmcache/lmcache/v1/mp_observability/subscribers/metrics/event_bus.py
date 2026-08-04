# SPDX-License-Identifier: Apache-2.0

"""EventBus self-metrics subscriber.

Surfaces two metrics describing the EventBus's own health (per #3108):

- ``lmcache_mp.event_bus.dropped_events_total`` (observable counter)
- ``lmcache_mp.event_bus.subscriber_exceptions`` (observable counter,
  attr ``subscriber_name``)

Unlike the other subscribers in this package, these metrics are not
driven by events — they observe bus state via the ``EventBus`` accessors
and report on every OTel scrape.  ``get_subscriptions`` therefore returns
an empty mapping.
"""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventCallback,
    EventSubscriber,
)


class EventBusSelfMetricsSubscriber(EventSubscriber):
    """Registers OTel observers for EventBus health metrics."""

    def __init__(self, bus: EventBus) -> None:
        meter = metrics.get_meter("lmcache.event_bus")

        meter.create_observable_counter(
            "lmcache_mp.event_bus.dropped_events_total",
            callbacks=[lambda _o: [metrics.Observation(bus.dropped_events_count())]],
            description=(
                "Cumulative events dropped because the EventBus queue was "
                "at max_queue_size."
            ),
        )
        meter.create_observable_counter(
            "lmcache_mp.event_bus.subscriber_exceptions",
            callbacks=[
                lambda _o: [
                    metrics.Observation(count, {"subscriber_name": name})
                    for name, count in bus.subscriber_exception_counts().items()
                ]
            ],
            description=(
                "Cumulative exceptions raised by subscriber callbacks "
                "during EventBus dispatch, tagged by ``subscriber_name``."
            ),
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {}
