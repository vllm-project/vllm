# SPDX-License-Identifier: Apache-2.0

"""L2 storage logging subscriber — debug logs for L2 store/prefetch events."""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)


class L2LoggingSubscriber(EventSubscriber):
    """Logs L2 store and prefetch events at debug level."""

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L2_STORE_SUBMITTED: self._on_store_submitted,
            EventType.L2_STORE_COMPLETED: self._on_store_completed,
            EventType.L2_PREFETCH_LOOKUP_SUBMITTED: self._on_lookup_submitted,
            EventType.L2_PREFETCH_LOOKUP_COMPLETED: self._on_lookup_completed,
            EventType.L2_PREFETCH_LOAD_SUBMITTED: self._on_load_submitted,
            EventType.L2_PREFETCH_LOAD_COMPLETED: self._on_load_completed,
            EventType.L2_KEYS_EVICTED: self._on_evicted,
        }

    def _on_store_submitted(self, event: Event) -> None:
        logger.debug(
            "L2 store submitted: %d keys to adapter %d",
            event.metadata["key_count"],
            event.metadata["adapter_index"],
        )

    def _on_store_completed(self, event: Event) -> None:
        logger.debug(
            "L2 store completed: adapter %d, %d succeeded, %d failed",
            event.metadata["adapter_index"],
            event.metadata["succeeded_count"],
            event.metadata["failed_count"],
        )

    def _on_lookup_submitted(self, event: Event) -> None:
        logger.debug(
            "L2 prefetch lookup submitted: request %d, %d keys to %d adapters",
            event.metadata["request_id"],
            event.metadata["key_count"],
            event.metadata["adapter_count"],
        )

    def _on_lookup_completed(self, event: Event) -> None:
        logger.debug(
            "L2 prefetch lookup completed: request %d, %d prefix hits",
            event.metadata["request_id"],
            event.metadata["prefix_hit_count"],
        )

    def _on_load_submitted(self, event: Event) -> None:
        logger.debug(
            "L2 prefetch load submitted: request %d, %d keys to %d adapters",
            event.metadata["request_id"],
            event.metadata["key_count"],
            event.metadata["adapter_count"],
        )

    def _on_load_completed(self, event: Event) -> None:
        logger.debug(
            "L2 prefetch load completed: request %d, %d loaded, %d failed",
            event.metadata["request_id"],
            event.metadata["loaded_count"],
            event.metadata["failed_count"],
        )

    def _on_evicted(self, event: Event) -> None:
        logger.debug(
            "L2 eviction: %d keys",
            event.metadata["key_count"],
        )
