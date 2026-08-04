# SPDX-License-Identifier: Apache-2.0

"""StorageManager logging subscriber — debug logs for SM events.

Logs are emitted via Python's standard logging module.  When OpenTelemetry
is installed, ``init_logger`` automatically attaches an OTel
``LoggingHandler`` so records are forwarded to OTel when a
``LoggerProvider`` is configured at startup.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)


class SMLoggingSubscriber(EventSubscriber):
    """Logs StorageManager events at debug level."""

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.SM_READ_PREFETCHED: self._on_read_prefetched,
            EventType.SM_READ_PREFETCHED_FINISHED: self._on_read_prefetched_finished,
            EventType.SM_WRITE_RESERVED: self._on_write_reserved,
            EventType.SM_WRITE_FINISHED: self._on_write_finished,
        }

    def _on_read_prefetched(self, event: Event) -> None:
        logger.debug(
            "SM read prefetched: %d succeeded, %d failed",
            len(event.metadata["succeeded_keys"]),
            len(event.metadata["failed_keys"]),
        )

    def _on_read_prefetched_finished(self, event: Event) -> None:
        logger.debug(
            "SM read prefetched finished: %d succeeded, %d failed",
            len(event.metadata["succeeded_keys"]),
            len(event.metadata["failed_keys"]),
        )

    def _on_write_reserved(self, event: Event) -> None:
        logger.debug(
            "SM write reserved: %d succeeded, %d failed",
            len(event.metadata["succeeded_keys"]),
            len(event.metadata["failed_keys"]),
        )

    def _on_write_finished(self, event: Event) -> None:
        logger.debug(
            "SM write finished: %d succeeded, %d failed",
            len(event.metadata["succeeded_keys"]),
            len(event.metadata["failed_keys"]),
        )
