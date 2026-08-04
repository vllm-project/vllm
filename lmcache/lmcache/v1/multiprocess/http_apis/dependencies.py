# SPDX-License-Identifier: Apache-2.0
"""HTTP-layer wiring for the MP server's cache handlers.

Holds the per-app context (the typed services the handlers operate on) and the
FastAPI dependency that retrieves it from the request. The context is built once
in the server lifespan (after the engine is initialized). Retrieving it is the
one genuinely HTTP-shaped concern here -- a request that races startup gets a
``503`` -- so it lives in this adapter layer, not in ``cache_control``.
"""

# Standard
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

# Third Party
from fastapi import HTTPException, Request

# First Party
from lmcache.v1.multiprocess.cache_control.object_service import ObjectService
from lmcache.v1.multiprocess.cache_control.prefetch_service import PrefetchService


@dataclass
class MPHTTPContext:
    """Typed collaborators the MP server's cache handlers operate on.

    Attributes:
        engine: The node's cache engine (``MPCacheServer``); the diagnostics
            handlers (clear / checksums) reach engine internals directly.
        object_service: Key-addressed L1/L2 delete, adapter / object listing.
        prefetch_service: Warm-prefetch submit / status (owns the job table).
    """

    engine: Any
    object_service: ObjectService
    prefetch_service: PrefetchService


def build_context(engine: Any) -> MPHTTPContext:
    """Construct the per-app context once the engine is ready.

    Called from the server lifespan; the result is stashed on
    ``app.state.context`` for :func:`get_context` to return.

    Args:
        engine: The initialized node cache engine.

    Returns:
        A fresh :class:`MPHTTPContext` wrapping the engine and its services.
    """
    return MPHTTPContext(
        engine=engine,
        object_service=ObjectService(engine),
        prefetch_service=PrefetchService(engine),
    )


def get_context(request: Request) -> MPHTTPContext:
    """Return the per-app :class:`MPHTTPContext`, or raise ``HTTPException``.

    Args:
        request: The FastAPI request whose ``app.state`` carries the context.

    Returns:
        The live :class:`MPHTTPContext`.

    Raises:
        HTTPException: 503 when the engine (and thus the context) isn't
            initialized yet -- e.g. a request that races server startup. This is
            a transport/DI concern, not a domain failure.
    """
    context = getattr(request.app.state, "context", None)
    if context is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="server not initialized"
        )
    return context
