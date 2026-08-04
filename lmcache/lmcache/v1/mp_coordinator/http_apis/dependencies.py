# SPDX-License-Identifier: Apache-2.0
"""Typed per-app context for the coordinator's HTTP handlers.

Instead of each handler reaching into ``request.app.state`` with stringly-typed
``getattr`` calls, handlers resolve a single :class:`CoordinatorContext` via
:func:`get_context`. The context is built in ``create_app``; ``outbound_client``
is filled in by the lifespan (it must bind to the running event loop), so a
dedicated accessor narrows that ``Optional`` for the dispatch handlers.
"""

# Standard
from dataclasses import dataclass

# Third Party
from fastapi import Request
import httpx

# First Party
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.cache_control.eviction_manager import L2EvictionManager
from lmcache.v1.mp_coordinator.cache_control.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.cache_control.usage_manager import L2UsageManager
from lmcache.v1.mp_coordinator.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.registry import InstanceRegistry
from lmcache.v1.multiprocess.token_hasher import TokenHasher


@dataclass
class CoordinatorContext:
    """Shared collaborators the coordinator's HTTP handlers operate on.

    Attributes:
        registry: Fleet membership (``MPInstance`` by ``instance_id``).
        quota_manager: Per-``cache_salt`` L2 quota state.
        usage_manager: Per-``cache_salt`` L2 usage tracking.
        eviction_manager: Quota/LRU L2 eviction dispatcher; also holds the L2
            pin set consulted when computing eviction plans.
        prefetch_manager: Warm-prefetch proxy to MP servers.
        token_hasher: Resolves a pin request's ``token_ids`` to object keys
            (configured to match the fleet's ``chunk_size`` / ``hash_algorithm``).
        key_directory: Fleet-wide key → placements directory built from
            MP-server cache events (eventually consistent).
        outbound_client: Shared async client for coordinator -> MP calls. Set by
            the lifespan (bound to the running loop); ``None`` until then.
    """

    registry: InstanceRegistry
    quota_manager: QuotaManager
    usage_manager: L2UsageManager
    eviction_manager: L2EvictionManager
    prefetch_manager: PrefetchManager
    token_hasher: TokenHasher
    key_directory: KeyDirectory
    outbound_client: httpx.AsyncClient | None = None


def get_context(request: Request) -> CoordinatorContext:
    """Return the per-app :class:`CoordinatorContext`.

    Args:
        request: The FastAPI request whose ``app.state`` carries the context.

    Returns:
        The shared :class:`CoordinatorContext`.

    Raises:
        RuntimeError: If the context is not initialized (wired by
            ``create_app``, so this should not happen in practice).
    """
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise RuntimeError("coordinator context not initialized")
    return ctx


def get_outbound_client(request: Request) -> httpx.AsyncClient:
    """Return the lifespan-bound outbound client, or raise if unset.

    Args:
        request: The FastAPI request.

    Returns:
        The shared outbound :class:`httpx.AsyncClient`.

    Raises:
        RuntimeError: If accessed before the lifespan filled it in (e.g. a bare
            app with no startup).
    """
    client = get_context(request).outbound_client
    if client is None:
        raise RuntimeError("outbound client not initialized")
    return client
