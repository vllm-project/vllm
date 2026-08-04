# SPDX-License-Identifier: Apache-2.0

"""Shared OTel span registry for cross-subscriber context propagation.

:class:`SpanRegistry` stores active spans keyed by ``(session_id, span_name)``
so that multiple subscribers sharing the same instance can look up parent
span contexts without tight coupling.

All reads and writes happen on the single EventBus drain thread — no locking
is needed.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class SpanRegistry:
    """Registry of active OTel spans, keyed by ``(session_id, span_name)``.

    ``span_name`` is a logical identifier (e.g. ``"request"``, ``"retrieve"``,
    ``"store"``) that decouples parent-lookup from the concrete OTel span name.

    All methods must be called from the EventBus drain thread; no locking is
    performed.

    Example usage across two subscribers that share one registry::

        registry = SpanRegistry()
        bus.register_subscriber(MPServerTracingSubscriber(registry))
        bus.register_subscriber(MyL1TracingSubscriber(registry))

        # Inside MyL1TracingSubscriber._on_l1_start:
        parent_ctx = registry.get_context(session_id, "retrieve") \\
                     or registry.get_context(session_id, "request")
        span = tracer.start_span("l1.read", context=parent_ctx, ...)
    """

    def __init__(self) -> None:
        # (session_id, span_name) -> (span, otel_context)
        self._active: dict[tuple[str, str], tuple[Any, Any]] = {}

    def open(self, session_id: str, span_name: str, span: Any, ctx: Any) -> None:
        """Register an open span and its OTel context.

        Args:
            session_id: The request session identifier.
            span_name: Logical name for the span (e.g. ``"request"``,
                ``"retrieve"``).
            span: The OTel span object.
            ctx: The OTel context containing the span (from
                ``trace.set_span_in_context(span)``).
        """
        self._active[(session_id, span_name)] = (span, ctx)

    def get(self, session_id: str, span_name: str) -> tuple[Any, Any] | None:
        """Return ``(span, context)`` for an open span, or ``None``.

        Args:
            session_id: The request session identifier.
            span_name: Logical span name to look up.

        Returns:
            ``(span, otel_context)`` if the span is registered, else ``None``.
        """
        return self._active.get((session_id, span_name))

    def get_context(self, session_id: str, span_name: str) -> Any | None:
        """Return the OTel context for an open span, or ``None``.

        Convenience wrapper over :meth:`get` for callers that only need the
        context to pass as a ``context=`` argument when starting a child span.

        Args:
            session_id: The request session identifier.
            span_name: Logical span name to look up.

        Returns:
            The OTel context, or ``None`` if the span is not registered.
        """
        entry = self._active.get((session_id, span_name))
        return entry[1] if entry is not None else None

    def pop(self, session_id: str, span_name: str) -> tuple[Any, Any] | None:
        """Remove and return ``(span, context)``, or ``None`` if absent.

        Args:
            session_id: The request session identifier.
            span_name: Logical span name to remove.

        Returns:
            ``(span, otel_context)`` if the span was registered, else ``None``.
        """
        return self._active.pop((session_id, span_name), None)

    def all_session_ids(self) -> set[str]:
        """Return the set of session IDs that have at least one open span.

        Returns:
            Set of session ID strings currently tracked in the registry.
        """
        return {k[0] for k in self._active}

    def clear_session(self, session_id: str) -> None:
        """End all open spans for *session_id* and remove them.

        Intended for use in ``shutdown()`` to clean up leaked spans when the
        EventBus stops unexpectedly.

        Args:
            session_id: The request session identifier to clean up.
        """
        keys = [k for k in self._active if k[0] == session_id]
        for k in keys:
            span, _ = self._active.pop(k)
            try:
                span.end()
            except Exception:
                logger.debug(
                    "SpanRegistry: error ending leaked span for session %s key %s",
                    session_id,
                    k[1],
                )


_registry: SpanRegistry | None = None


def get_span_registry() -> SpanRegistry:
    """Return the process-level singleton :class:`SpanRegistry`.

    The registry is created on first call.  Pass a fresh :class:`SpanRegistry`
    directly to subscriber constructors when test isolation is needed.

    Returns:
        The shared :class:`SpanRegistry` instance.
    """
    global _registry
    if _registry is None:
        _registry = SpanRegistry()
    return _registry
