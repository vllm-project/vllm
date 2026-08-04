# SPDX-License-Identifier: Apache-2.0
"""
BaseServiceFactory: Abstract interface for creating LMCache service components.

Each serving engine integration (e.g., vLLM) should implement a concrete
ServiceFactory that determines which components to create for each role.
"""

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.health_monitor.base import HealthMonitor
from lmcache.v1.health_monitor.constants import (
    DEFAULT_PING_INTERVAL,
    PING_INTERVAL_CONFIG_KEY,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.observability import PrometheusLogger
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.internal_api_server.api_server import InternalAPIServer
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer
    from lmcache.v1.manager import LMCacheManager
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
    from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher

logger = init_logger(__name__)


class BaseServiceFactory(ABC):
    """Abstract base for creating LMCache service components.

    Subclasses must implement all methods to provide the appropriate
    components for their serving engine integration.
    """

    @abstractmethod
    def get_engine_instance_id(self) -> str:
        """Return the instance_id used to register the engine with
        LMCacheEngineBuilder. Used by LMCacheManager for engine destruction."""
        raise NotImplementedError

    @abstractmethod
    def get_or_create_metadata(self) -> Optional["LMCacheMetadata"]:
        raise NotImplementedError

    @abstractmethod
    def get_or_create_lmcache_engine(self) -> Optional["LMCacheEngine"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_lookup_client(self) -> Optional["LookupClientInterface"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_prometheus_logger(self) -> Optional["PrometheusLogger"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_lookup_server(
        self,
    ) -> Optional[Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_offload_server(self) -> Optional["ZMQOffloadServer"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_runtime_plugin_launcher(
        self,
    ) -> Optional["RuntimePluginLauncher"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_internal_api_server(
        self, lmcache_manager: "LMCacheManager"
    ) -> Optional["InternalAPIServer"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_health_monitor(
        self, lmcache_manager: "LMCacheManager"
    ) -> Optional[HealthMonitor]:
        raise NotImplementedError

    def _create_health_monitor(
        self,
        lmcache_manager: "LMCacheManager",
        config: LMCacheEngineConfig,
        engine: Optional["LMCacheEngine"] = None,
    ) -> HealthMonitor:
        """Create, configure, and start the health monitor.

        Shared implementation used by subclass maybe_create_health_monitor.
        """
        # First Party
        from lmcache.observability import PrometheusLogger
        from lmcache.v1.periodic_thread import (
            PeriodicThreadRegistry,
            ThreadLevel,
        )

        ping_interval = config.get_extra_config_value(
            PING_INTERVAL_CONFIG_KEY, DEFAULT_PING_INTERVAL
        )
        health_monitor = HealthMonitor(
            manager=lmcache_manager,
            ping_interval=ping_interval,
        )

        if engine is not None:
            engine.set_health_monitor(health_monitor)

        health_monitor.start()
        logger.info("Health monitor initialized and started")

        metadata = lmcache_manager.lmcache_engine_metadata
        if metadata is not None:
            prometheus_logger = PrometheusLogger.GetOrCreate(
                metadata,
                config=config,
            )
            prometheus_logger.lmcache_is_healthy.set_function(
                lambda: 1 if lmcache_manager.is_healthy() else 0
            )

            registry = PeriodicThreadRegistry.get_instance()

            prometheus_logger.periodic_threads_total_count.set_function(
                lambda: len(registry.get_all())
            )
            prometheus_logger.periodic_threads_running_count.set_function(
                lambda: registry.get_running_count()
            )
            prometheus_logger.periodic_threads_active_count.set_function(
                lambda: registry.get_active_count()
            )

            for level in ThreadLevel:
                level_name = level.value
                total_attr = f"periodic_threads_{level_name}_total"
                running_attr = f"periodic_threads_{level_name}_running"
                active_attr = f"periodic_threads_{level_name}_active"

                if hasattr(prometheus_logger, total_attr):
                    getattr(prometheus_logger, total_attr).set_function(
                        lambda lvl=level: registry.get_count_by_level(lvl)["total"]
                    )
                if hasattr(prometheus_logger, running_attr):
                    getattr(prometheus_logger, running_attr).set_function(
                        lambda lvl=level: registry.get_count_by_level(lvl)["running"]
                    )
                if hasattr(prometheus_logger, active_attr):
                    getattr(prometheus_logger, active_attr).set_function(
                        lambda lvl=level: registry.get_count_by_level(lvl)["active"]
                    )

        return health_monitor
