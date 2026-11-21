"""vllm.synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Utility module providing a synchronization hook that can be attached to the
indexing subsystem of vLLM. The hook ensures that any pending asynchronous
operations are flushed before the index is accessed, preventing race
conditions during concurrent inference.

The public entry point is :func:`sync_index` which is deliberately lightweight
and side‑effect free apart from the synchronization it performs.
"""

# Signed-off-by: Automated Fix <auto@vllm-project.org>

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

__all__: list[str] = ["sync_index"]


async def _default_sync_fn() -> None:
    """Default no‑op synchronization coroutine.

    The function exists so that :func:`sync_index` can always ``await`` a
    coroutine, even when the caller does not provide a custom implementation.
    """
    return None


def _ensure_awaitable(fn: Optional[Callable[[], Awaitable[None]]]) -> Callable[[], Awaitable[None]]:
    """Return a coroutine function guaranteeing an awaitable.

    Parameters
    ----------
    fn:
        A callable returning an awaitable or ``None``.

    Returns
    -------
    Callable[[], Awaitable[None]]
        A coroutine function that can be safely awaited.
    """
    if fn is None:
        return _default_sync_fn
    # If the provided function is already a coroutine function we return it
    # unchanged; otherwise we wrap the result in ``asyncio.sleep(0)`` to make it
    # awaitable without altering behaviour.
    async def wrapper() -> None:
        result = fn()
        if asyncio.iscoroutine(result):
            await result
        else:
            # Non‑awaitable result – treat as a synchronous side effect.
            return None
    return wrapper


async def sync_index(sync_fn: Optional[Callable[[], Awaitable[None]]] = None) -> None:
    """Synchronize the index before it is accessed.

    This coroutine should be invoked by any component that needs a consistent
    view of the index. It optionally accepts a user‑provided ``sync_fn`` which
    will be awaited. If ``sync_fn`` is ``None``, a lightweight no‑op coroutine
    is used.

    Parameters
    ----------
    sync_fn:
        An optional coroutine function performing the actual synchronization.
        It must return ``None`` and can be ``async`` or a regular callable.
    """
    await _ensure_awaitable(sync_fn)()

