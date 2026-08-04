# SPDX-License-Identifier: Apache-2.0
"""FastAPI application factory for the mp coordinator.

The coordinator is a FastAPI app. Endpoints are auto-discovered from the
``http_apis`` package (the same convention as the mp server's HTTP API) and stay
thin, operating on the shared collaborators carried on ``app.state``: ``config``,
``registry``, ``quota_manager``, ``usage_manager``, and ``eviction_manager``.
The lifespan runs background tasks for health-checking (eviction of instances
whose heartbeats have lapsed) and L2 eviction (quota enforcement).

Adding a capability = a new ``http_apis/<name>_api.py`` router (auto-discovered)
that uses those shared collaborators. To push to an mp server, a future router
resolves the instance's address from the registry (``ip`` + ``http_port``) and
POSTs to that server's specific endpoint. A domain with real logic/state of its
own adds a ``<name>_service.py`` stashed on ``app.state`` here; thin domains
(like membership) just use the registry directly.
"""

# Standard
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import contextlib

# Third Party
from fastapi import FastAPI
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.blend_directory import GlobalBlendMatcher
from lmcache.v1.mp_coordinator.cache_control.eviction_manager import (
    L2EvictionManager,
)
from lmcache.v1.mp_coordinator.cache_control.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.cache_control.resync_manager import L2ResyncManager
from lmcache.v1.mp_coordinator.cache_control.usage_manager import L2UsageManager
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.http_apis.dependencies import CoordinatorContext
from lmcache.v1.mp_coordinator.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.registry import InstanceRegistry
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.utils.router_discovery import discover_api_routers

logger = init_logger(__name__)


def evict_stale(registry: InstanceRegistry, instance_timeout: float) -> list[str]:
    """Deregister every instance whose heartbeat is older than the timeout.

    Args:
        registry: The shared instance registry.
        instance_timeout: Max seconds since the last heartbeat before eviction.

    Returns:
        The ids of instances evicted in this sweep.
    """
    evicted = []
    for instance_id in registry.stale(instance_timeout):
        if registry.deregister(instance_id) is not None:
            logger.warning("Instance %s timed out; evicted", instance_id)
            evicted.append(instance_id)
    return evicted


def create_app(config: MPCoordinatorConfig) -> FastAPI:
    """Build the coordinator FastAPI app.

    Args:
        config: The coordinator configuration.

    Returns:
        A configured FastAPI application. ``app.state`` carries the shared
        collaborators (``config``, ``registry``, ``quota_manager``,
        ``usage_manager``, ``key_directory``, ``blend_directory``); all
        ``http_apis`` routers are registered.
    """
    registry = InstanceRegistry()
    quota_manager = QuotaManager()
    usage_manager = L2UsageManager()
    eviction_manager = L2EvictionManager(
        quota_manager=quota_manager,
        usage_manager=usage_manager,
        eviction_ratio=config.eviction_ratio,
        trigger_watermark=config.trigger_watermark,
    )
    resync_manager = L2ResyncManager(
        usage_manager=usage_manager,
        eviction_manager=eviction_manager,
        page_size=config.resync_page_size,
    )
    prefetch_manager = PrefetchManager()
    # Resolves pin requests' token_ids to object keys; must match the fleet's
    # chunk size and hash algorithm (see MPCoordinatorConfig).
    token_hasher = TokenHasher(
        chunk_size=config.chunk_size, hash_algorithm=config.hash_algorithm
    )
    blend_directory = GlobalBlendMatcher(
        chunk_size=config.chunk_size, probe_stride=config.blend_probe_stride
    )
    key_directory = KeyDirectory()
    # Typed context the cache handlers resolve via ``get_context``;
    # ``outbound_client`` is filled in by the lifespan (bound to the loop).
    ctx = CoordinatorContext(
        registry=registry,
        quota_manager=quota_manager,
        usage_manager=usage_manager,
        eviction_manager=eviction_manager,
        prefetch_manager=prefetch_manager,
        token_hasher=token_hasher,
        key_directory=key_directory,
    )

    async def _health_loop() -> None:
        """Evict stale instances on a timer until cancelled."""
        while True:
            await asyncio.sleep(config.health_check_interval)
            evict_stale(registry, config.instance_timeout)

    async def _eviction_loop(http_client: httpx.AsyncClient) -> None:
        """Periodically check usage against quotas and dispatch
        eviction RPCs to any one registered MP server.

        Safe to start immediately: salts without an explicit quota are
        exempt from eviction until the external quota controller sets a
        default limit via ``PUT /quota/config`` (after re-syncing the
        per-salt quotas), so a cold quota table cannot mass-evict."""
        while True:
            await asyncio.sleep(config.eviction_check_interval)
            await eviction_manager.execute_evictions(registry, http_client)

    async def _startup_resync(http_client: httpx.AsyncClient) -> None:
        """One-shot backfill of usage + eviction trackers from a live
        MP server's actual L2 contents."""
        await resync_manager.wait_and_resync(
            registry=registry,
            http_client=http_client,
            poll_interval=config.resync_poll_interval,
            max_wait=config.resync_max_wait,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Start background tasks and clean up resources on shutdown."""
        # Shared async client for outbound coordinator → MP server
        # calls (eviction dispatch + startup resync). Created inside
        # the lifespan so it binds to the running event loop.
        outbound_client = httpx.AsyncClient(timeout=30.0)
        # Fill in the context's outbound client so request handlers (e.g. POST
        # /cache/prefetches) can issue outbound calls; background loops capture
        # the local ``outbound_client`` directly.
        ctx.outbound_client = outbound_client
        health_task = None
        eviction_task = None
        resync_task = None
        if config.health_check_interval > 0:
            health_task = asyncio.create_task(_health_loop())
        if config.eviction_check_interval > 0:
            eviction_task = asyncio.create_task(_eviction_loop(outbound_client))
        if config.enable_startup_resync:
            resync_task = asyncio.create_task(_startup_resync(outbound_client))
        logger.info(
            "MP coordinator listening on http://%s:%d", config.host, config.port
        )
        try:
            yield
        finally:
            for task in (health_task, eviction_task, resync_task):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            await eviction_manager.wait_for_in_flight_dispatches()
            await outbound_client.aclose()

    app = FastAPI(title="LMCache MP Coordinator", version="1.0.0", lifespan=lifespan)
    # The typed context carries the cache collaborators handlers compose from
    # (registry, quota/usage/eviction/prefetch managers, outbound client);
    # handlers resolve it via ``get_context``.
    app.state.ctx = ctx
    # Out-of-context collaborators kept on app.state directly: ``config`` and the
    # blend directory (its own router), plus ``resync_manager`` for the lifespan.
    app.state.config = config
    app.state.resync_manager = resync_manager
    app.state.blend_directory = blend_directory

    apis_path = Path(__file__).parent / "http_apis"
    package = f"{__package__}.http_apis"
    for router in discover_api_routers(apis_path, package):
        app.include_router(router)

    return app
