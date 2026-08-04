# SPDX-License-Identifier: Apache-2.0

"""Observability-aware timeout error.

Raise :class:`LMCacheTimeoutError` instead of the builtin ``TimeoutError`` so
each timeout is reported to MP observability. The ``ban-raw-timeout-error``
pre-commit hook enforces this across ``lmcache/``.
"""

# Future
from __future__ import annotations

# Standard
import traceback

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    get_event_bus,
    is_observability_enabled,
)

logger = init_logger(__name__)


class LMCacheTimeoutError(TimeoutError):
    """``TimeoutError`` that publishes a ``TIMEOUT_RAISED`` event on construction.

    Subclasses the builtin, so existing ``except TimeoutError`` handlers still
    catch it. Publishing is skipped (no event, no stack capture) when
    observability is disabled, and a publish failure never propagates out of
    ``__init__``.

    Args:
        message: Timeout description; also the event's ``message`` field.
        session_id: Request id for correlating the timeout with its originating
            request; empty when unavailable.
    """

    def __init__(self, message: str, *, session_id: str = "") -> None:
        super().__init__(message)
        if not is_observability_enabled():
            return
        # Stack at the raise site, minus this __init__ frame.
        stacktrace = "".join(traceback.format_stack()[:-1])
        self._publish_timeout_event(message, stacktrace, session_id)

    def _publish_timeout_event(
        self, message: str, stacktrace: str, session_id: str
    ) -> None:
        """Publish a ``TIMEOUT_RAISED`` event; swallow any failure."""
        try:
            get_event_bus().publish(
                Event(
                    event_type=EventType.TIMEOUT_RAISED,
                    metadata={
                        "message": message,
                        "exception_type": type(self).__name__,
                        "stacktrace": stacktrace,
                    },
                    session_id=session_id,
                )
            )
        except Exception:
            logger.debug("Failed to publish TIMEOUT_RAISED event", exc_info=True)
