# SPDX-License-Identifier: Apache-2.0

"""Dispatcher mapping recorded ``qualname`` strings to live callables.

The recorder writes one :class:`~lmcache.v1.mp_observability.trace.format.Record`
per decorated call, tagged by the function's fully-qualified name.
The dispatcher translates those strings back into concrete calls on a
live :class:`~lmcache.v1.distributed.storage_manager.StorageManager`.

Adding support for a new traced operation is a two-line change: put
``@enable_tracing`` on the function, then register a handler here with
a matching ``qualname``.  No per-op schemas, no new event types.
"""

# Future
from __future__ import annotations

# Standard
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_manager import StorageManager

logger = init_logger(__name__)

#: Fully-qualified name of the ``StorageManager`` class.  Used to
#: build the qualnames of all its traced methods.  Kept as a constant
#: so tests and dispatcher registrations stay in lock-step if the class
#: is ever renamed or moved.
_SM_PREFIX = "lmcache.v1.distributed.storage_manager.StorageManager"


@dataclass
class ReplayContext:
    """State carried across dispatcher invocations during one replay.

    The driver creates exactly one ``ReplayContext`` per trace file,
    hands it to every dispatched handler, and closes the StorageManager
    when replay finishes.

    Attributes:
        sm: The live StorageManager that receives replayed calls.
        open_read_contexts: FIFO queue of
            ``read_prefetched_results`` contexts entered but not yet
            exited.  Matching ``__enter__``/``__exit__`` records are
            popped in the order they were entered for the same
            ``keys`` tuple.  A dict-of-deques keyed on
            ``tuple(keys)`` supports interleaved contexts across
            different key sets.
    """

    sm: StorageManager
    open_read_contexts: dict[tuple[ObjectKey, ...], deque[AbstractContextManager]] = (
        field(default_factory=dict)
    )


#: Type of a dispatcher handler: takes a :class:`ReplayContext` and an
#: already-decoded ``args`` dict (keys = parameter names, values =
#: native Python values restored by
#: :mod:`lmcache.v1.mp_observability.trace.codecs`).  Handlers return
#: nothing; any return value from the live call is discarded.
Handler = Callable[[ReplayContext, dict[str, Any]], None]


class CallDispatcher:
    """Registry mapping recorded qualnames to replay handlers.

    Handlers are plain callables — no per-op subclass hierarchy.  The
    default factory :func:`build_default_dispatcher` populates the
    registry with every v1 ``StorageManager`` operation.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, qualname: str, handler: Handler) -> None:
        """Register a handler for *qualname*.

        Args:
            qualname: Fully-qualified call-site name exactly matching
                the ``qualname`` field written by the recorder.
            handler: Callable invoked on each matching record.

        Raises:
            ValueError: If a handler is already registered for
                *qualname*.
        """
        if qualname in self._handlers:
            raise ValueError(f"handler already registered for {qualname!r}")
        self._handlers[qualname] = handler

    def has(self, qualname: str) -> bool:
        """Return ``True`` if a handler is registered for *qualname*."""
        return qualname in self._handlers

    def registered_qualnames(self) -> list[str]:
        """Return the list of currently registered qualnames.

        Returns:
            A new list of qualname strings, suitable for inspection or
            assertion in tests.
        """
        return list(self._handlers)

    def dispatch(
        self,
        qualname: str,
        context: ReplayContext,
        args: dict[str, Any],
    ) -> None:
        """Invoke the handler for *qualname*.

        Args:
            qualname: The recorded qualname.
            context: Replay context passed through to the handler.
            args: Decoded argument dict for the call.

        Raises:
            KeyError: If no handler is registered for *qualname*.  The
                driver catches this and logs a warning so unknown
                qualnames (e.g. from a future trace level) do not stop
                replay.
        """
        handler = self._handlers.get(qualname)
        if handler is None:
            raise KeyError(qualname)
        handler(context, args)


# ---------------------------------------------------------------------------
# Default handlers for v1 StorageManager operations
# ---------------------------------------------------------------------------


def _call_sm_method(method_name: str) -> Handler:
    """Build a handler that forwards a record to ``sm.<method_name>``.

    The returned callable invokes ``getattr(ctx.sm, method_name)(**args)``
    and discards the result.  Used for every "plain" traced method on
    StorageManager — ``reserve_write``, ``finish_write``,
    ``submit_prefetch_task``, ``finish_read_prefetched``.

    Args:
        method_name: Attribute name on the live StorageManager.

    Returns:
        A :data:`Handler` closure.
    """

    def _handler(ctx: ReplayContext, args: dict[str, Any]) -> None:
        method = getattr(ctx.sm, method_name)
        method(**args)

    _handler.__name__ = f"_call_sm_{method_name}"
    return _handler


def _enter_read_prefetched(ctx: ReplayContext, args: dict[str, Any]) -> None:
    """Handle a ``read_prefetched_results.__enter__`` record.

    Enters the live context manager and stashes it under
    ``tuple(keys)``.  The matching ``__exit__`` handler pops the top
    entry for that key tuple, preserving FIFO order when multiple
    contexts are simultaneously open for identical key lists.

    Args:
        ctx: Active replay context.
        args: Decoded record arguments.  Must contain ``"keys"``.
    """
    keys = args["keys"]
    cm = ctx.sm.read_prefetched_results(keys)
    cm.__enter__()
    key_tuple = tuple(keys)
    ctx.open_read_contexts.setdefault(key_tuple, deque()).append(cm)


def _exit_read_prefetched(ctx: ReplayContext, args: dict[str, Any]) -> None:
    """Handle a ``read_prefetched_results.__exit__`` record.

    Pops and exits the oldest context opened for ``tuple(keys)``.  If
    no matching open context exists — typically because the trace was
    truncated between the enter and exit events — logs a warning and
    continues so replay does not abort.

    Args:
        ctx: Active replay context.
        args: Decoded record arguments.  Must contain ``"keys"``.
    """
    keys = args["keys"]
    key_tuple = tuple(keys)
    pending = ctx.open_read_contexts.get(key_tuple)
    if not pending:
        logger.warning(
            "trace replay: read_prefetched_results.__exit__ with no "
            "matching __enter__ (keys=%d); ignoring",
            len(keys),
        )
        return
    cm = pending.popleft()
    if not pending:
        del ctx.open_read_contexts[key_tuple]
    # Pass a clean exit — replay does not reproduce caller-side
    # exceptions.  The context manager's ``finally`` block runs
    # regardless, so read locks are released.
    cm.__exit__(None, None, None)


def build_default_dispatcher() -> CallDispatcher:
    """Return a :class:`CallDispatcher` populated with v1 handlers.

    Covers every qualname the storage-level recorder emits:

    * ``StorageManager.reserve_write``
    * ``StorageManager.finish_write``
    * ``StorageManager.submit_prefetch_task``
    * ``StorageManager.finish_read_prefetched``
    * ``StorageManager.read_prefetched_results.__enter__``
    * ``StorageManager.read_prefetched_results.__exit__``

    Returns:
        A ready-to-use dispatcher.  Callers may further register
        additional handlers on it for future trace levels.
    """
    dispatcher = CallDispatcher()
    for method_name in (
        "reserve_write",
        "finish_write",
        "submit_prefetch_task",
        "finish_read_prefetched",
    ):
        dispatcher.register(
            f"{_SM_PREFIX}.{method_name}",
            _call_sm_method(method_name),
        )
    dispatcher.register(
        f"{_SM_PREFIX}.read_prefetched_results.__enter__",
        _enter_read_prefetched,
    )
    dispatcher.register(
        f"{_SM_PREFIX}.read_prefetched_results.__exit__",
        _exit_read_prefetched,
    )
    return dispatcher
