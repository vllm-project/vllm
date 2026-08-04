# SPDX-License-Identifier: Apache-2.0
"""
StandaloneServiceFactory: Service factory for LMCache standalone mode.

Creates LMCache service components without vLLM dependencies.
"""

# Standard
from typing import TYPE_CHECKING, Any, Callable, Optional

# First Party
from lmcache.integration.base_service_factory import BaseServiceFactory
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.health_monitor.base import HealthMonitor
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.metadata import LMCacheMetadata

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.manager import LMCacheManager


class StandaloneServiceFactory(BaseServiceFactory):
    """Service factory for standalone LMCache mode (no vLLM)."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        gpu_connector: Any,
        broadcast_fn: Callable,
        broadcast_object_fn: Callable,
    ):
        self._config = config
        self._metadata = metadata
        self._gpu_connector = gpu_connector
        self._broadcast_fn = broadcast_fn
        self._broadcast_object_fn = broadcast_object_fn
        self._engine: Optional[LMCacheEngine] = None

    def get_engine_instance_id(self) -> str:
        return self._config.lmcache_instance_id

    def get_or_create_metadata(self) -> Optional[LMCacheMetadata]:
        return self._metadata

    def get_or_create_lmcache_engine(self) -> Optional[LMCacheEngine]:
        if self._engine is not None:
            return self._engine

        instance_id = self._config.lmcache_instance_id
        self._engine = LMCacheEngineBuilder.get_or_create(
            instance_id=instance_id,
            config=self._config,
            metadata=self._metadata,
            gpu_connector=self._gpu_connector,
            broadcast_fn=self._broadcast_fn,
            broadcast_object_fn=self._broadcast_object_fn,
        )
        return self._engine

    def maybe_create_lookup_client(self):
        return None

    def maybe_create_prometheus_logger(self):
        return None

    def maybe_create_lookup_server(self):
        return None

    def maybe_create_offload_server(self):
        return None

    def maybe_create_runtime_plugin_launcher(self):
        return None

    def maybe_create_internal_api_server(
        self, lmcache_manager: "LMCacheManager"
    ) -> Optional[InternalAPIServer]:
        return InternalAPIServer(lmcache_manager)

    def maybe_create_health_monitor(
        self, lmcache_manager: "LMCacheManager"
    ) -> Optional[HealthMonitor]:
        return self._create_health_monitor(lmcache_manager, self._config, self._engine)
