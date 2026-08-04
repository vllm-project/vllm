# SPDX-License-Identifier: Apache-2.0
"""
LMCacheManager: A unified manager for LMCache internal components.

This module provides a clean interface to manage LMCache components lifecycle,
decoupling the vLLM adapter from internal LMCache implementation details.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import TYPE_CHECKING, Any, Optional, Union
import time
import traceback

# Third Party
import torch

# First Party
from lmcache.integration.base_service_factory import BaseServiceFactory
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.health_monitor.base import HealthMonitor
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer

logger = init_logger(__name__)


class LMCacheManager:
    """
    LMCacheManager manages the lifecycle of LMCache internal components.

    For an integration to utilize the Manager, define a ServiceFactory
    that determines which components to create for which workers.

    This class encapsulates the initialization and shutdown of:
    - LMCacheEngine
    - LookupClient / LookupServer
    - OffloadServer
    - InternalAPIServer
    - RuntimePluginLauncher
    - HealthMonitor
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        service_factory: BaseServiceFactory,
        connector: Optional[Any] = None,
    ):
        """
        Initialize LMCacheManager.

        Args:
            config: LMCache engine configuration
            service_factory: Factory for creating service components
            role: The role string ("scheduler" or "worker")
            connector: Reference to LMCacheConnectorV1Impl for internal
                       API server
        """
        self._config = config
        self._service_factory = service_factory
        self._connector: Any = connector

        # Flag to track if initialization failed
        self._init_failed = False
        self._init_failed_reason: str = ""

        self._health_monitor: Optional[HealthMonitor] = None
        self._lmcache_engine_metadata: Optional[LMCacheMetadata] = None
        self._lmcache_engine: Optional[LMCacheEngine] = None
        self._lookup_client: Optional[LookupClientInterface] = None
        self._lookup_server: Optional[
            Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]
        ] = None
        self._offload_server: Optional[ZMQOffloadServer] = None
        self._runtime_plugin_launcher: Optional[RuntimePluginLauncher] = None
        self._api_server: Optional[InternalAPIServer] = None

        # Initialize components via service factory
        try:
            self._lmcache_engine_metadata = service_factory.get_or_create_metadata()
            self._lmcache_engine = service_factory.get_or_create_lmcache_engine()
            self._lookup_client = service_factory.maybe_create_lookup_client()
            self._lookup_server = service_factory.maybe_create_lookup_server()
            self._offload_server = service_factory.maybe_create_offload_server()
            self._runtime_plugin_launcher = (
                service_factory.maybe_create_runtime_plugin_launcher()
            )
            self._api_server = service_factory.maybe_create_internal_api_server(
                lmcache_manager=self
            )
        except Exception as e:
            self._init_failed = True
            self._init_failed_reason = "".join(traceback.format_exception(e))
            logger.error(
                "Failed to initialize LMCacheManager components: %s. "
                "System will operate in degraded mode (recompute).",
                self._init_failed_reason,
            )

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance."""
        return self._lmcache_engine

    @property
    def lmcache_engine_metadata(self) -> Optional[LMCacheMetadata]:
        """Get the LMCache engine metadata."""
        return self._lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional[LookupClientInterface]:
        """Get the lookup client instance."""
        return self._lookup_client

    @property
    def lookup_server(
        self,
    ) -> Optional[Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]]:
        """Get the lookup server instance."""
        return self._lookup_server

    @property
    def offload_server(self) -> Optional[ZMQOffloadServer]:
        """Get the offload server instance."""
        return self._offload_server

    @property
    def api_server(self) -> Optional[InternalAPIServer]:
        """Get the API server instance."""
        return self._api_server

    @property
    def health_monitor(self) -> Optional[HealthMonitor]:
        """Get the health monitor instance."""
        return self._health_monitor

    @property
    def kv_caches(self) -> dict[str, torch.Tensor]:
        if self._connector is not None and hasattr(self._connector, "kv_caches"):
            return self._connector.kv_caches
        return {}

    @property
    def config(self) -> LMCacheEngineConfig:
        """Get the LMCache engine configuration."""
        return self._config

    # ==================== Lifecycle Methods ====================

    def start_services(self) -> None:
        """
        Start all managed services.

        Managed services include:
        - InternalAPIServer: HTTP server exposing internal APIs for
          monitoring and management (e.g., cache stats, flush operations).
        - RuntimePluginLauncher: Launches external plugin processes defined
          in the configuration (e.g., custom telemetry, cache warming).
        """
        if self._api_server is not None:
            self._api_server.start()

        if self._runtime_plugin_launcher is not None:
            self._runtime_plugin_launcher.launch_plugins()

    def post_init(self) -> None:
        """
        Post-initialization after KV caches are registered.
        """
        # If initialization already failed, mark engine and return early
        if self._init_failed:
            if self._lmcache_engine is not None:
                self._lmcache_engine.mark_init_failed(self._init_failed_reason)
            logger.warning("Skipping post_init due to previous initialization failure")
            return

        if self._lmcache_engine is None:
            # Initialize health monitor for scheduler (even without engine)
            self._init_health_monitor()
            return

        try:
            # First Party
            from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
                LMCacheAsyncLookupServer,
            )

            async_lookup_server = None
            if self._config.enable_async_loading and self._lookup_server is not None:
                assert isinstance(self._lookup_server, LMCacheAsyncLookupServer)
                async_lookup_server = self._lookup_server

            self._lmcache_engine.post_init(async_lookup_server=async_lookup_server)

            # Initialize health monitor after engine post_init completes
            self._init_health_monitor()
        except Exception as e:
            self._handle_post_init_failure(e)

    def _init_health_monitor(self) -> None:
        """Initialize the health monitor via the service factory.

        Called during post_init after all components are initialized.
        """
        self._health_monitor = self._service_factory.maybe_create_health_monitor(
            lmcache_manager=self
        )

    def _handle_post_init_failure(self, e: Exception) -> None:
        """
        Handle initialization failure during post_init.

        Args:
            e: The exception that caused the failure
        """
        self._init_failed = True
        self._init_failed_reason = "".join(traceback.format_exception(e))
        if self._lmcache_engine is not None:
            self._lmcache_engine.mark_init_failed(self._init_failed_reason)
        logger.error(
            "Failed during post_init: %s. "
            "System will operate in degraded mode (recompute).",
            self._init_failed_reason,
        )

    def stop_services(self) -> None:
        """Stop all managed components gracefully."""
        logger.info("Stopping LMCacheManager services...")
        start_time = time.time()
        errors: list[tuple[str, Union[str, Exception]]] = []

        def _safe_close(name: str, close_fn, timeout: float = 10.0):
            """Helper to close a resource with timeout protection."""
            try:
                logger.info("Closing %s...", name)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(close_fn)
                    try:
                        future.result(timeout=timeout)
                        logger.info("%s closed successfully", name)
                    except TimeoutError:
                        logger.error(
                            "%s close operation timed out after %ss. "
                            "Continuing with shutdown...",
                            name,
                            timeout,
                        )
                        errors.append((name, "Timeout"))
            except Exception as e:
                logger.error("Error closing %s: %s", name, e)
                errors.append((name, e))

        # Stop health monitor first
        if self._health_monitor is not None:
            _safe_close("health_monitor", self._health_monitor.stop, timeout=5.0)

        # Close offload server
        if self._offload_server is not None:
            _safe_close("offload_server", self._offload_server.close, timeout=10.0)

        # Stop plugins
        if self._runtime_plugin_launcher is not None:
            _safe_close(
                "runtime_plugin_launcher",
                self._runtime_plugin_launcher.stop_plugins,
                timeout=10.0,
            )

        # Stop API server
        if self._api_server is not None:
            _safe_close("api_server", self._api_server.stop, timeout=10.0)

        # Close lookup server
        if self._lookup_server is not None:
            _safe_close("lookup_server", self._lookup_server.close, timeout=10.0)

        # Close lookup client
        if self._lookup_client is not None:
            _safe_close("lookup_client", self._lookup_client.close, timeout=10.0)

        # Destroy cache engine
        try:
            engine_instance_id = self._service_factory.get_engine_instance_id()
            logger.info("Destroying LMCache engine: %s", engine_instance_id)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    LMCacheEngineBuilder.destroy, engine_instance_id
                )
                try:
                    future.result(timeout=15.0)
                    logger.info("LMCache engine destroyed successfully")
                except TimeoutError:
                    logger.error(
                        "Cache engine destroy timed out after 15s. "
                        "Continuing with shutdown..."
                    )
                    errors.append(("cache_engine", "Timeout"))
        except Exception as e:
            logger.error("Error destroying cache engine: %s", e)
            errors.append(("cache_engine", e))

        elapsed = time.time() - start_time
        if errors:
            logger.warning(
                "Shutdown completed with %d errors in %.2fs: %s",
                len(errors),
                elapsed,
                errors,
            )
        else:
            logger.info(
                "LMCacheManager services stopped successfully in %.2fs",
                elapsed,
            )

    # ==================== Lookup Management ====================

    def close_lookup_client(self) -> dict:
        """
        Close the current lookup client.

        Returns:
            dict: Result with old client type info.
        """
        old_type = self._get_lookup_type_str(self._lookup_client)
        if self._lookup_client is not None:
            try:
                self._lookup_client.close()
                self._lookup_client = None
                logger.info("Closed lookup client: %s", old_type)
            except Exception as e:
                logger.warning("Error closing lookup client: %s", e)
        return {"old": old_type}

    def create_lookup_client(self, dryrun: bool = False) -> dict:
        """
        Create a new lookup client using the current configuration.

        Args:
            dryrun: If True, only return what would be created without
                actually creating it.

        Returns:
            dict: Result with new client type info.
        """
        # First Party
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        if self._lmcache_engine_metadata is None:
            return {"error": "metadata not available"}

        if dryrun:
            client = LookupClientFactory.create_lookup_client(
                self._config,
                self._lmcache_engine_metadata,
                self._lmcache_engine,
            )
            new_type = self._get_lookup_type_str(client)
            client.close()
            return {"new": new_type, "dryrun": True}

        self._lookup_client = LookupClientFactory.create_lookup_client(
            self._config,
            self._lmcache_engine_metadata,
            self._lmcache_engine,
        )
        new_type = self._get_lookup_type_str(self._lookup_client)
        logger.info("Created lookup client: %s", new_type)
        return {"new": new_type}

    def recreate_lookup_client(self) -> dict:
        """
        Recreate the lookup client (close + create).

        Returns:
            dict: Result with old and new client type info.
        """
        if self._lookup_client is None:
            return {"error": "lookup client not available"}
        result = self.close_lookup_client()
        create_result = self.create_lookup_client()
        result.update(create_result)
        return result

    def close_lookup_server(self) -> dict:
        """
        Close the current lookup server.

        Returns:
            dict: Result with old server type info.
        """
        old_type = self._get_lookup_type_str(self._lookup_server)
        if self._lookup_server is not None:
            try:
                self._lookup_server.close()
                self._lookup_server = None
                logger.info("Closed lookup server: %s", old_type)
            except Exception as e:
                logger.warning("Error closing lookup server: %s", e)
        return {"old": old_type}

    def create_lookup_server(self, dryrun: bool = False) -> dict:
        """
        Create a new lookup server using the current configuration.

        Args:
            dryrun: If True, only return what would be created without
                actually creating it.

        Returns:
            dict: Result with new server type info.
        """
        # First Party
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        if self._lmcache_engine is None:
            return {"error": "engine not available"}
        if self._lmcache_engine_metadata is None:
            return {"error": "metadata not available"}

        if dryrun:
            server = LookupClientFactory.create_lookup_server(
                self._lmcache_engine,
                self._lmcache_engine_metadata,
            )
            new_type = self._get_lookup_type_str(server)
            if server is not None:
                server.close()
            return {"new": new_type, "dryrun": True}

        self._lookup_server = LookupClientFactory.create_lookup_server(
            self._lmcache_engine,
            self._lmcache_engine_metadata,
        )
        new_type = self._get_lookup_type_str(self._lookup_server)
        logger.info("Created lookup server: %s", new_type)
        return {"new": new_type}

    def recreate_lookup_server(self) -> dict:
        """
        Recreate the lookup server (close + create).

        Returns:
            dict: Result with old and new server type info.
        """
        if self._lookup_server is None:
            return {"error": "lookup server not available"}

        result = self.close_lookup_server()
        create_result = self.create_lookup_server()
        result.update(create_result)
        return result

    def _get_lookup_type_str(self, obj) -> str:
        """
        Get type string for lookup client/server, including wrapper info.

        Returns format: "OuterWrapper(InnerWrapper(CoreType))" or "None"
        """
        if obj is None:
            return "None"

        parts = []
        current = obj
        while True:
            parts.append(type(current).__name__)
            if hasattr(current, "actual_lookup_client"):
                current = current.actual_lookup_client
            else:
                break

        if len(parts) == 1:
            return parts[0]
        result = parts[-1]
        for wrapper in reversed(parts[:-1]):
            result = "%s(%s)" % (wrapper, result)
        return result

    def get_lookup_info(self) -> dict:
        """
        Get information about the current lookup client and server.

        Returns:
            dict: Information about lookup client/server types and role.
        """
        return {
            "client": self._get_lookup_type_str(self._lookup_client),
            "server": self._get_lookup_type_str(self._lookup_server),
        }

    # ==================== Health & Info ====================

    def is_healthy(self) -> bool:
        """
        Check if the LMCacheManager is healthy.

        Returns False if:
        - Initialization failed
        - HealthMonitor reports unhealthy
        - Engine reports unhealthy

        Returns:
            bool: True if healthy, False otherwise
        """
        if self._init_failed:
            return False
        if self._lmcache_engine is not None and not self._lmcache_engine.is_healthy():
            return False
        if self._health_monitor is not None:
            return self._health_monitor.is_healthy()
        return True

    def get_inference_info(self) -> dict:
        """Get inference information by delegating to the connector.

        Returns:
            dict: Dictionary containing inference information,
                  or empty dict if connector is not available.
        """
        if self._connector is not None and hasattr(
            self._connector, "get_inference_info"
        ):
            return self._connector.get_inference_info()
        return {}
