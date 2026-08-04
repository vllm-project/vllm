# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
import argparse
import asyncio
import contextlib
import time

# Third Party
from fastapi import FastAPI
import httpx
import uvicorn

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    l1_exposes_single_memory_region,
    parse_args_to_config,
)
from lmcache.v1.mp_coordinator.cache_control.event_listener import L2EventListener
from lmcache.v1.mp_coordinator.cache_events import (
    CacheEventSubscriber,
    HttpCacheEventSink,
)
from lmcache.v1.mp_coordinator.registrar import keep_registered
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    add_observability_args,
    parse_args_to_observability_config,
)
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.multiprocess.config import (
    DEFAULT_COORDINATOR_CONFIG,
    CoordinatorConfig,
    HTTPFrontendConfig,
    MPServerConfig,
    add_coordinator_args,
    add_http_frontend_args,
    add_mp_server_args,
    add_p2p_args,
    parse_args_to_coordinator_config,
    parse_args_to_http_frontend_config,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.http_api_registry import (
    HTTPAPIRegistry,
)
from lmcache.v1.multiprocess.http_apis.dependencies import build_context
from lmcache.v1.multiprocess.http_apis.error_handlers import register_error_handlers
from lmcache.v1.multiprocess.mp_runtime_plugin_launcher import (
    MPRuntimePluginLauncher,
)
from lmcache.v1.multiprocess.server import run_cache_server

logger = init_logger(__name__)


# Module-level config holders, set by run_http_server() before FastAPI startup.
# Stored in a dict so the lifespan closure captures the mutable container.
_configs: dict = {}


# ----------------------------
# FastAPI lifespan for initialization and cleanup
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifecycle of the LMCache HTTP server.

    On startup: Initialize ZMQ server and cache engine.
    On shutdown: Clean up ZMQ server and cache engine resources.
    """
    # Startup
    logger.info(
        "Starting LMCache HTTP server... (accelerator available: %s)",
        torch_dev.is_available(),
    )
    mp_config = _configs["mp"]
    coordinator_config = _configs.get("coordinator") or DEFAULT_COORDINATOR_CONFIG

    result = run_cache_server(
        mp_config=mp_config,
        storage_manager_config=_configs["storage_manager"],
        obs_config=_configs["observability"],
        return_engine=True,
        start_prometheus_http_server=False,
        coordinator_config=coordinator_config,
    )
    assert result is not None, "run_cache_server returned None with return_engine=True"
    zmq_server, engine = result

    # Launch runtime plugins if configured. Plugins receive the full
    # server config (including HTTP host/port) via the
    # LMCACHE_RUNTIME_PLUGIN_CONFIG environment variable.
    plugin_launcher = None
    if mp_config.runtime_plugin_config.locations:
        extra_kwargs = {}
        http_config = _configs.get("http")
        if http_config is not None:
            extra_kwargs["http_config"] = http_config
        plugin_launcher = MPRuntimePluginLauncher(
            runtime_plugin_config=mp_config.runtime_plugin_config,
            mp_config=mp_config,
            storage_manager_config=_configs["storage_manager"],
            obs_config=_configs["observability"],
            **extra_kwargs,
        )
        plugin_launcher.launch_plugins()

    app.state.zmq_server = zmq_server
    app.state.engine = engine
    # Typed per-app context the cache handlers resolve via ``get_context``
    # (built now that the engine is ready).
    app.state.context = build_context(engine)
    app.state.plugin_launcher = plugin_launcher

    # Optionally register this server with an MP coordinator (enabled when
    # coordinator config has a URL). A generic HTTP client sends the requests;
    # the keep_registered task registers, heartbeats, and deregisters on
    # shutdown. Best-effort: failures are logged and retried, never fatal.
    http_config = _configs.get("http")
    coordinator_client = None
    coordinator_registration_task = None
    if coordinator_config.url and http_config is not None:
        coordinator_client = httpx.AsyncClient(timeout=10.0)
        # Canonical id resolved by run_cache_server above; shared with
        # the OTel service.instance.id so membership matches metrics/traces.
        coordinator_registration_task = asyncio.create_task(
            keep_registered(
                coordinator_client,
                coordinator_config.url,
                http_port=http_config.http_port,
                instance_id=mp_config.instance_id,
                advertise_ip=coordinator_config.advertise_ip,
                heartbeat_interval=coordinator_config.heartbeat_interval,
                p2p_advertised_url=mp_config.p2p_config.advertise_url,
                mq_port=mp_config.port if mp_config.p2p_config.enabled else 0,
            )
        )
    # Optionally report cache events to the coordinator (one flag gates
    # both streams until they are unified): the legacy L2 usage stream
    # (/quota/events) for quota tracking and eviction, and the key
    # directory stream for fleet-wide placement tracking.
    coordinator_l2_event_client = None
    coordinator_l2_event_task = None
    if (
        coordinator_client is not None
        and coordinator_config is not None
        and coordinator_config.url
        and coordinator_config.event_reporting
    ):
        coordinator_l2_event_client = L2EventListener(
            coordinator_client,
            coordinator_config.url,
            instance_id=mp_config.instance_id,
            flush_interval=coordinator_config.event_flush_interval,
        )
        if engine.storage_manager is not None:
            engine.storage_manager.register_l2_listener(coordinator_l2_event_client)
        coordinator_l2_event_task = asyncio.create_task(
            coordinator_l2_event_client.run()
        )

    # Coordinator event stream
    if (
        coordinator_client is not None
        and coordinator_config.url
        and coordinator_config.event_reporting
    ):
        get_event_bus().register_subscriber(
            CacheEventSubscriber(
                sink=HttpCacheEventSink(coordinator_config.url),
                instance_id=mp_config.instance_id,
                # Server start time: fences out placements this instance
                # reported before a restart (its pools restarted empty).
                incarnation=int(time.time()),
                flush_interval=coordinator_config.event_flush_interval,
            )
        )

    app.state.coordinator_client = coordinator_client
    app.state.coordinator_registration_task = coordinator_registration_task
    app.state.coordinator_l2_event_task = coordinator_l2_event_task

    logger.info("LMCache HTTP server initialized")

    yield

    # Shutdown
    logger.info("Shutting down LMCache HTTP server...")
    coordinator_l2_event_task = getattr(app.state, "coordinator_l2_event_task", None)
    if coordinator_l2_event_task is not None:
        coordinator_l2_event_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_l2_event_task
    coordinator_registration_task = getattr(
        app.state, "coordinator_registration_task", None
    )
    if coordinator_registration_task is not None:
        coordinator_registration_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_registration_task
    coordinator_client = getattr(app.state, "coordinator_client", None)
    if coordinator_client is not None:
        await coordinator_client.aclose()
    launcher = getattr(app.state, "plugin_launcher", None)
    if launcher is not None:
        launcher.stop_plugins()
    get_event_bus().stop()
    if hasattr(app.state, "zmq_server") and app.state.zmq_server is not None:
        app.state.zmq_server.close()
    engine.close()
    logger.info("LMCache HTTP server stopped")


app = FastAPI(title="LMCache HTTP API", version="1.0.0", lifespan=lifespan)

# Automatically discover and register all HTTP API endpoints
registry = HTTPAPIRegistry(app)
registry.register_all_apis()

# Map cache-control domain errors to HTTP responses centrally, so routes need
# no try/except (auto-discovery finds routers, not exception handlers).
register_error_handlers(app)


def run_http_server(
    http_config: HTTPFrontendConfig,
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    obs_config: ObservabilityConfig,
    coordinator_config: CoordinatorConfig,
) -> None:
    """
    Run the LMCache HTTP server with integrated MP (ZMQ) server.

    Args:
        http_config: Configuration for the HTTP frontend
        mp_config: Configuration for the ZMQ multiprocess server
        storage_manager_config: Configuration for the storage manager
        obs_config: Configuration for the observability stack
        coordinator_config: Configuration for MP coordinator registration
            (an empty URL disables registration)

    Raises:
        ValueError: If P2P is enabled without a coordinator URL, or with an L1
            tier that is not a single registerable memory region; or if
            coordinator event reporting is enabled with observability
            disabled (the cache-event stream rides the event bus).
    """
    if mp_config.p2p_config.enabled:
        if not coordinator_config.url:
            raise ValueError(
                "P2P requires a coordinator for peer discovery: set "
                "--coordinator-url (or LMCACHE_COORDINATOR_URL) when "
                "--p2p-advertise-url is set."
            )
        if not l1_exposes_single_memory_region(storage_manager_config):
            raise ValueError(
                "P2P requires a single L1 memory region the transfer channel "
                "can register; it is incompatible with GDS L1 (--gds-l1-path) "
                "and Device-DAX L1 (--l1-devdax-path)."
            )
    if coordinator_config.event_reporting and not obs_config.enabled:
        raise ValueError(
            "--coordinator-event-reporting rides the observability event "
            "bus: remove --disable-observability to report cache events."
        )
    _configs["mp"] = mp_config
    _configs["storage_manager"] = storage_manager_config
    _configs["observability"] = obs_config
    _configs["http"] = http_config
    _configs["coordinator"] = coordinator_config
    app.state.configs = _configs

    config = uvicorn.Config(
        app=app,
        host=http_config.http_host,
        port=http_config.http_port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(config)

    logger.info(
        "Starting LMCache HTTP server on http://%s:%d",
        http_config.http_host,
        http_config.http_port,
    )
    server.run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMCache HTTP Server with integrated MP Cache Server"
    )
    add_http_frontend_args(parser)
    add_mp_server_args(parser)
    add_p2p_args(parser)
    add_storage_manager_args(parser)
    add_observability_args(parser)
    add_coordinator_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    http_config = parse_args_to_http_frontend_config(args)
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    obs_config = parse_args_to_observability_config(args)
    coordinator_config = parse_args_to_coordinator_config(args)
    run_http_server(
        http_config=http_config,
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
        coordinator_config=coordinator_config,
    )
