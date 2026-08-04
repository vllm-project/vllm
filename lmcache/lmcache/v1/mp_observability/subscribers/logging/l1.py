# SPDX-License-Identifier: Apache-2.0

"""L1 logging subscriber — debug logs for L1Manager events.

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


class L1LoggingSubscriber(EventSubscriber):
    """Logs L1Manager events at debug level."""

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L1_READ_RESERVED: self._on_read_reserved,
            EventType.L1_READ_FINISHED: self._on_read_finished,
            EventType.L1_WRITE_RESERVED: self._on_write_reserved,
            EventType.L1_WRITE_FINISHED: self._on_write_finished,
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED: (
                self._on_write_finished_and_read_reserved
            ),
            EventType.L1_KEYS_EVICTED: self._on_evicted,
        }

    def _on_read_reserved(self, event: Event) -> None:
        logger.debug("L1 read reserved: %d keys", len(event.metadata["keys"]))

    def _on_read_finished(self, event: Event) -> None:
        logger.debug("L1 read finished: %d keys", len(event.metadata["keys"]))

    def _on_write_reserved(self, event: Event) -> None:
        logger.debug("L1 write reserved: %d keys", len(event.metadata["keys"]))

    def _on_write_finished(self, event: Event) -> None:
        logger.debug("L1 write finished: %d keys", len(event.metadata["keys"]))

    def _on_write_finished_and_read_reserved(self, event: Event) -> None:
        logger.debug(
            "L1 write finished and read reserved: %d keys",
            len(event.metadata["keys"]),
        )

    def _on_evicted(self, event: Event) -> None:
        logger.debug("L1 eviction: %d keys", len(event.metadata["keys"]))
