# SPDX-License-Identifier: Apache-2.0
"""MPCacheServer compositor and unified cache server entry point."""

# Standard
import argparse
import shutil
import signal
import sys
import time

# Third Party
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.usage_telemetry.mp import InitializeMPUsageContext
from lmcache.usage_telemetry.mp_continuous import InitializeMPContinuousUsage
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    add_observability_args,
    init_observability,
    parse_args_to_observability_config,
)
from lmcache.v1.mp_observability.gc_monitor import (
    init_gc_monitor,
    shutdown_gc_monitor,
)
from lmcache.v1.mp_observability.trace import maybe_initialize_trace_recorder
from lmcache.v1.multiprocess.config import (
    DEFAULT_COORDINATOR_CONFIG,
    CoordinatorConfig,
    MPServerConfig,
    add_coordinator_args,
    add_mp_server_args,
    parse_args_to_coordinator_config,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    EngineModule,
    HandlerSpec,
    InstanceLivenessTarget,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.experimental import EXPERIMENTAL_TRANSFER
from lmcache.v1.multiprocess.modules.experimental.qstore import QStoreModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext

logger = init_logger(__name__)


class MPCacheServer:
    """Compositor that assembles pluggable engine modules.

    Holds the shared :class:`MPCacheServerContext` and a list of
    :class:`EngineModule` instances.  Provides aggregated
    ``report_status()`` and ``close()`` across all modules.

    Args:
        context: The shared engine context.
        modules: List of engine modules to compose.
    """

    def __init__(
        self,
        context: MPCacheServerContext,
        modules: list[EngineModule],
    ) -> None:
        self._context = context
        self._modules = modules

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context."""
        return self._context

    def report_status(self) -> dict:
        """Return an aggregated status dict from all modules.

        Returns:
            Combined status from the storage manager, engine metadata,
            and each module's ``report_status()`` output.
        """
        sm = self._context.storage_manager.report_status()
        status: dict = {
            "is_healthy": sm["is_healthy"],
            "engine_type": self.__class__.__name__,
            "chunk_size": self._context.chunk_size,
            "hash_algorithm": self._context.token_hasher.hash_algorithm_name,
            "active_sessions": self._context.session_manager.active_count(),
            "storage_manager": sm,
        }
        for module in self._modules:
            status.update(module.report_status())
        return status

    def close(self) -> None:
        """Close all modules and release shared resources."""
        for module in self._modules:
            module.close()
        self._context.close()
        logger.info("MPCacheServer closed")

    # HTTP-layer passthroughs lost in the engine refactor.

    @property
    def storage_manager(self) -> StorageManager:
        """Used by ``/quota/*``."""
        return self._context.storage_manager

    @property
    def cache_contexts(self) -> dict[int, BaseCacheContext] | None:
        """Used by ``/cache/checksums``; unwraps :class:`ContextEntry`."""
        for module in self._modules:
            if isinstance(module, LMCacheDrivenTransferModule):
                return {
                    i: e.cache_context
                    for i, e in module.context_entries_snapshot().items()
                }
        return None

    def clear(self) -> None:
        """Used by ``/cache/clear``; delegates to :class:`ManagementModule`."""
        for module in self._modules:
            if isinstance(module, ManagementModule):
                module.clear()
                return
        raise RuntimeError("MPCacheServer.clear: no ManagementModule registered")


def add_handler_helper(
    server: MessageQueueServer, request_type: RequestType, handler_function
):
    """Register a handler with the message queue server.

    Args:
        server: The message queue server.
        request_type: The request type to handle.
        handler_function: The handler callable.
    """
    payload_classes = get_payload_classes(request_type)
    handler_type = get_handler_type(request_type)
    server.add_handler(
        request_type,
        payload_classes,
        handler_type,
        handler_function,
    )


def _build_modules(
    ctx: MPCacheServerContext,
    mp_config: MPServerConfig,
    coordinator_config: CoordinatorConfig,
) -> list[EngineModule]:
    """Assemble the list of engine modules based on configuration.

    Args:
        ctx: The shared engine context.
        mp_config: Server configuration determining which modules to load.
        coordinator_config: Coordinator connection used by the P2P controller
            for peer discovery.

    Returns:
        List of initialized engine modules.

    Raises:
        ValueError: If blend engine is requested with
        supported_transfer_mode="engine_driven".
    """
    lookup_module = LookupModule(ctx)
    p2p_controller = P2PController(
        ctx,
        mp_config.p2p_config,
        coordinator_config,
        mp_config.instance_id,
    )

    # Build the transfer and blend modules first so the ManagementModule can
    # be constructed with them as liveness targets / reap listeners. They are
    # the InstanceLivenessTargets the reaper scans.
    transfer_modules: list[EngineModule] = []
    if mp_config.supported_transfer_mode == "lmcache_driven":
        transfer_modules.append(LMCacheDrivenTransferModule(ctx))
    elif mp_config.supported_transfer_mode == "engine_driven":
        transfer_modules.append(EngineDrivenTransferModule(ctx))
    elif mp_config.supported_transfer_mode == "auto":
        transfer_modules.append(LMCacheDrivenTransferModule(ctx))
        transfer_modules.append(EngineDrivenTransferModule(ctx))
    else:
        raise ValueError(
            f"Unsupported supported_transfer_mode '{mp_config.supported_transfer_mode}'"
        )

    logger.info("Supported transfer mode: %s", mp_config.supported_transfer_mode)

    # Targets the reaper scans (and reap-notifies). The transfer modules own
    # per-instance liveness; BlendV3Module is appended below as a state mirror.
    liveness_targets: list[InstanceLivenessTarget] = [
        m
        for m in transfer_modules
        if isinstance(m, (LMCacheDrivenTransferModule, EngineDrivenTransferModule))
    ]

    # At most one blend module is ever built (engine_type selects one).
    blend_module: EngineModule | None = None

    if mp_config.engine_type == "blend_legacy":
        if mp_config.supported_transfer_mode == "engine_driven":
            raise ValueError(
                "Legacy blend engine requires supported_transfer_mode to be "
                f"'lmcache_driven' or 'auto', got "
                f"'{mp_config.supported_transfer_mode}'"
            )
        # First Party
        from lmcache.v1.multiprocess.modules.blend import BlendModule

        blend_module = BlendModule(ctx)

    # "blend" selects CacheBlend V3 (the current implementation).
    if mp_config.engine_type == "blend":
        if mp_config.supported_transfer_mode == "engine_driven":
            raise ValueError(
                "blend (V3) engine requires supported_transfer_mode "
                f"'lmcache_driven' or 'auto', got "
                f"'{mp_config.supported_transfer_mode}'"
            )
        # First Party
        from lmcache.v1.mp_coordinator.blend_client import (
            BlendCoordinatorClient,
        )
        from lmcache.v1.multiprocess.modules.blend_v3 import BlendV3Module

        transfer_module = next(
            m for m in transfer_modules if isinstance(m, LMCacheDrivenTransferModule)
        )
        # Opt-in: enabled when a coordinator URL is configured (flag or
        # LMCACHE_COORDINATOR_URL, resolved at config parsing); otherwise
        # None and the blend module matches purely locally.
        coordinator = BlendCoordinatorClient.maybe_create(coordinator_config.url)
        blend_v3 = BlendV3Module(
            ctx,
            transfer_module,
            coordinator=coordinator,
            enable_segmented_prefix=mp_config.enable_segmented_prefix,
        )
        blend_module = blend_v3
        # blend_v3 mirrors per-instance CB rope state, so the reaper must
        # notify it via drop_instance_state when an instance is reaped.
        liveness_targets.append(blend_v3)

    # Experimental intermediate tensor transfer modules
    lmcache_driven_module = next(
        (m for m in transfer_modules if isinstance(m, LMCacheDrivenTransferModule)),
        None,
    )
    enabled_modules = set(mp_config.enable)
    experimental_transfer: list[str] = []
    experimental_modules: list[EngineModule] = []
    for enabled_module in enabled_modules:
        if enabled_module not in EXPERIMENTAL_TRANSFER:
            raise ValueError(
                f"Unknown --enable experimental module '{enabled_module}'."
            )
        if lmcache_driven_module is None:
            raise ValueError(
                f"Experimental module '{enabled_module}' requires "
                "supported_transfer_mode='lmcache_driven' or 'auto'."
            )
        module = QStoreModule(ctx)
        experimental_modules.append(module)
        liveness_targets.append(module)
        experimental_transfer.append(enabled_module)

    management = ManagementModule(
        ctx,
        liveness_targets=liveness_targets,
        worker_reap_timeout_seconds=mp_config.worker_reap_timeout_seconds,
        worker_registration_grace_seconds=mp_config.worker_registration_grace_seconds,
        experimental_transfer=experimental_transfer,
    )

    # ManagementModule precedes the transfer/blend modules so close() stops
    # and joins the reaper before those modules clear their state and before
    # storage_manager.close() runs.
    blend_modules = [blend_module] if blend_module is not None else []
    return [
        lookup_module,
        p2p_controller,
        management,
        *transfer_modules,
        *experimental_modules,
        *blend_modules,
    ]


def run_cache_server(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    obs_config: ObservabilityConfig,
    return_engine: bool = False,
    start_prometheus_http_server: bool = True,
    coordinator_config: CoordinatorConfig = DEFAULT_COORDINATOR_CONFIG,
) -> tuple[MessageQueueServer, MPCacheServer] | None:
    """Run the LMCache cache server with ZMQ message queue.

    Args:
        mp_config: Configuration for the ZMQ multiprocess server.
        storage_manager_config: Configuration for the storage manager.
        obs_config: Configuration for the observability stack.
        coordinator_config: Coordinator connection used by the P2P controller
            for peer discovery.
        return_engine: If True, return (server, engine) after starting;
                       if False, run blocking loop to keep server alive.
        start_prometheus_http_server: Whether to start a standalone
            Prometheus HTTP server in a background thread.  Set to
            ``False`` when an external HTTP framework already serves
            ``/metrics`` to avoid port conflicts or redundant servers.

    Returns:
        If return_engine is True: tuple of (MessageQueueServer, MPCacheServer).
        If return_engine is False: None (blocks until interrupted).
    """
    # mp_config.instance_id is this server's single source of identity (set via
    # --instance-id, else a random UUID v4). Project it onto the OTel
    # service.instance.id unless observability set that attribute explicitly, so
    # metrics/traces and coordinator membership all key on the same id.
    if obs_config.service_instance_id is None:
        obs_config.service_instance_id = mp_config.instance_id

    event_bus = init_observability(
        obs_config, start_prometheus_http_server=start_prometheus_http_server
    )

    init_gc_monitor(obs_config.gc_monitor)

    maybe_initialize_trace_recorder(event_bus, obs_config, storage_manager_config)

    # When the engine-driven path is loaded (auto or engine_driven):
    # apply shm_name from mp_config and verify capacity.
    if mp_config.supported_transfer_mode != "lmcache_driven":
        mem_cfg = storage_manager_config.l1_manager_config.memory_config
        if mp_config.shm_name is not None:
            mem_cfg.shm_name = mp_config.shm_name
        if mem_cfg.shm_name and sys.platform.startswith("linux"):
            logger.info("Checking if shm capacity is larger than L1 request")
            try:
                free_bytes = shutil.disk_usage("/dev/shm").free
                if free_bytes < mem_cfg.size_in_bytes:
                    logger.warning(
                        "Insufficient /dev/shm capacity: need %d bytes, have %d bytes. "
                        "Disabling SHM, falling back to pickle.",
                        mem_cfg.size_in_bytes,
                        free_bytes,
                    )
                    mem_cfg.shm_name = ""
            except OSError:
                logger.warning(
                    "Cannot verify /dev/shm capacity; disabling SHM.",
                    exc_info=True,
                )
                mem_cfg.shm_name = ""

    # blend engine: single object group + full per-chunk SWA KV
    is_blend = mp_config.engine_type == "blend"

    ctx = MPCacheServerContext(
        storage_manager_config=storage_manager_config,
        chunk_size=mp_config.chunk_size,
        hash_algorithm=mp_config.hash_algorithm,
        separate_object_groups=mp_config.separate_object_groups and not is_blend,
        full_sw_kv=is_blend,
    )

    modules = _build_modules(ctx, mp_config, coordinator_config)
    engine = MPCacheServer(ctx, modules)

    InitializeMPUsageContext(mp_config, storage_manager_config)
    InitializeMPContinuousUsage(event_bus, mp_config.chunk_size)

    zmq_context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{mp_config.host}:{mp_config.port}",
        context=zmq_context,
    )

    all_specs: list[HandlerSpec] = []
    for module in modules:
        all_specs.extend(module.get_handlers())

    for spec in all_specs:
        add_handler_helper(server, spec.request_type, spec.handler)

    affinity_types = [
        s.request_type for s in all_specs if s.pool == ThreadPoolType.AFFINITY
    ]
    normal_types = [
        s.request_type for s in all_specs if s.pool == ThreadPoolType.NORMAL
    ]
    if affinity_types:
        server.add_affinity_thread_pool(
            affinity_types, max_workers=mp_config.max_gpu_workers
        )
    if normal_types:
        server.add_normal_thread_pool(
            normal_types, max_workers=mp_config.max_cpu_workers
        )

    logger.info(
        "LMCache ZMQ cache server is running on tcp://%s:%d",
        mp_config.host,
        mp_config.port,
    )

    if not hasattr(torch_dev, "init"):
        logger.warning(
            "Backend '%s' does not support init(), skipping device init",
            torch_device_type,
        )
    else:
        torch_dev.init()
    server.start()

    logger.info("LMCache cache server is running...")

    if return_engine:
        return server, engine

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        event_bus.stop()
        server.close()
        engine.close()
    finally:
        shutdown_gc_monitor()
    return None


def parse_args():
    """Parse command line arguments for the cache server.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="LMCache ZMQ Cache Server (without HTTP)"
    )
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    add_observability_args(parser)
    add_coordinator_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    args = parse_args()
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    obs_config = parse_args_to_observability_config(args)
    coordinator_config = parse_args_to_coordinator_config(args)
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
        coordinator_config=coordinator_config,
    )
