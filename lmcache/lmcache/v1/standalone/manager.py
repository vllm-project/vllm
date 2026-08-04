# SPDX-License-Identifier: Apache-2.0
"""
StandaloneLMCacheManager: A specialized manager for LMCache standalone mode.

Uses a StandaloneServiceFactory to handle standalone mode specifically,
removing vLLM dependencies and simplifying the initialization logic.
"""

# Standard
from typing import Any, Callable, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.manager import LMCacheManager
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.standalone.standalone_service_factory import StandaloneServiceFactory

logger = init_logger(__name__)


class StandaloneLMCacheManager(LMCacheManager):
    """
    LMCacheManager specialized for standalone mode.

    Uses StandaloneServiceFactory to create components without
    vLLM dependencies.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        gpu_connector: Any,
        broadcast_fn: Callable,
        broadcast_object_fn: Callable,
        connector: Optional[Any] = None,
    ):
        """
        Initialize StandaloneLMCacheManager.

        Args:
            config: LMCache engine configuration
            metadata: Pre-constructed LMCacheMetadata
            gpu_connector: GPU connector instance
            broadcast_fn: Broadcast function for tensor parallel
            broadcast_object_fn: Broadcast function for objects
            connector: Reference to connector for internal API server
        """
        service_factory = StandaloneServiceFactory(
            config=config,
            metadata=metadata,
            gpu_connector=gpu_connector,
            broadcast_fn=broadcast_fn,
            broadcast_object_fn=broadcast_object_fn,
        )

        super().__init__(
            config=config,
            service_factory=service_factory,
            connector=connector,
        )

    def post_init(self) -> None:
        """Post-initialization for standalone mode."""
        if self._init_failed:
            if self._lmcache_engine is not None:
                self._lmcache_engine.mark_init_failed(self._init_failed_reason)
            logger.warning("Skipping post_init due to previous initialization failure")
            return

        try:
            if self._lmcache_engine is not None:
                # Standalone mode post-init (no async_lookup_server)
                self._lmcache_engine.post_init()

            # Initialize health monitor after engine post_init
            self._init_health_monitor()
        except Exception as e:
            self._handle_post_init_failure(e)
