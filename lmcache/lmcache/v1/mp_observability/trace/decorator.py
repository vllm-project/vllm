# SPDX-License-Identifier: Apache-2.0

"""``@enable_tracing`` decorator for capturing function calls.

The decorator publishes a single :data:`EventType.TRACE_CALL` event on
**function entry** (inputs only).  Output values and exceptions are not
captured — replay re-runs the function and observes the live outcome.

The decorator imposes near-zero overhead when tracing is disabled: a
single boolean attribute load is added to each call.  Argument
introspection is performed only when the gate is on.

Codecs that turn LMCache-specific argument types into msgpack-friendly
forms live in :mod:`lmcache.v1.mp_observability.trace.codecs`.  The
decorator deliberately does not import them; raw Python values are
attached to the event and the recorder encodes at write time.  This
keeps the decorator import-cheap and breaks an otherwise circular
dependency (``StorageManager → decorator → codecs → StorageManager``).
"""

# Future
from __future__ import annotations

# Standard
from functools import wraps
from typing import Any, Callable, Sequence, TypeVar
import inspect
import time

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

F = TypeVar("F", bound=Callable[..., Any])

# Module-level gate.  Flipped on by the trace recorder when it
# registers, off when it shuts down.  A simple bool is sufficient
# (tracing capture is single-process; mutual visibility across threads
# is not required for correctness — at worst a few events are missed
# during the toggle window).
_tracing_enabled: bool = False


def is_tracing_enabled() -> bool:
    """Return whether the trace gate is currently on."""
    return _tracing_enabled


def set_tracing_enabled(enabled: bool) -> None:
    """Flip the trace gate.

    Called by trace recorders during ``__init__`` (on) and ``close()``
    (off).  Direct callers should not normally use this.
    """
    global _tracing_enabled
    _tracing_enabled = enabled


def publish_call_event(qualname: str, args: dict[str, Any]) -> None:
    """Publish one ``TRACE_CALL`` event.

    Used by :func:`enable_tracing` and by manual instrumentation
    points (e.g. context-manager enter/exit) that cannot be wrapped
    by the decorator.

    Args:
        qualname: Fully-qualified name of the call site.
        args: Mapping of argument name to raw Python value.  Codec
            encoding happens later, in the recorder.

    ``time.monotonic()`` is sampled **here** (not on the drain thread)
    so the recorded ``t_mono`` aligns with ``Event.timestamp`` —
    otherwise the two clocks would drift by however long the drain
    lagged behind the publisher.
    """
    if not _tracing_enabled:
        return
    t_mono = time.monotonic()
    bus = get_event_bus()
    bus.publish(
        Event(
            event_type=EventType.TRACE_CALL,
            metadata={"qualname": qualname, "args": args, "t_mono": t_mono},
        )
    )


def enable_tracing(
    qualname: str | None = None,
    capture: Sequence[str] | None = None,
    redact: Sequence[str] = (),
) -> Callable[[F], F]:
    """Decorate a function so its calls publish ``TRACE_CALL`` events.

    Args:
        qualname: Fully-qualified call-site name placed in the event
            metadata.  Defaults to ``f"{func.__module__}.{func.__qualname__}"``.
        capture: If given, only these argument names are recorded.
            ``None`` means capture every parameter except ``self`` and
            ``cls``.
        redact: Argument names that must not be recorded.  Applied
            after ``capture``.

    Returns:
        A decorator that wraps the target function.

    The signature is bound once at decoration time via
    :func:`inspect.signature`, so per-call overhead is limited to a
    bool check (when disabled) or a ``Signature.bind_partial`` plus
    dict-comprehension (when enabled).
    """
    redact_set = frozenset(redact)
    capture_set = frozenset(capture) if capture is not None else None

    def deco(func: F) -> F:
        sig = inspect.signature(func)
        resolved_qualname = qualname or f"{func.__module__}.{func.__qualname__}"

        # Pre-compute parameter names to record.  ``self`` and ``cls``
        # are always dropped — recording the receiver object yields no
        # useful information for replay and would force a codec for
        # every receiver type.
        param_names = [
            name
            for name in sig.parameters
            if name not in ("self", "cls")
            and (capture_set is None or name in capture_set)
            and name not in redact_set
        ]
        param_set = frozenset(param_names)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _tracing_enabled:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                payload = {k: v for k, v in bound.arguments.items() if k in param_set}
                publish_call_event(resolved_qualname, payload)
            return func(*args, **kwargs)

        # Expose the resolved metadata for tests and dispatcher
        # registration.
        wrapper.__lmc_trace_qualname__ = resolved_qualname  # type: ignore[attr-defined]
        wrapper.__lmc_trace_params__ = tuple(param_names)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return deco
