# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.chunk_statistics_lookup_client import (
    ChunkStatisticsLookupClient,
)
from lmcache.v1.lookup_client.hit_limit_lookup_client import HitLimitLookupClient
from lmcache.v1.lookup_client.lmcache_lookup_client_bypass import (
    LMCacheBypassLookupClient,
)
from lmcache.v1.lookup_client.mooncake_lookup_client import MooncakeLookupClient
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc.zmq_transport import (
    SocketParams,
    ZmqReqRepClientTransport,
    ZmqRouterServerTransport,
)
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer

logger = init_logger(__name__)


class LookupClientFactory:
    """Factory for creating lookup clients and servers based on configuration."""

    @staticmethod
    def create_lookup_client(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        lmcache_engine: Optional[LMCacheEngine] = None,
    ) -> LookupClientInterface:
        """
        Create a lookup client based on the configuration.

        Args:
            config: The LMCache engine configuration
            metadata: The LMCache engine metadata (includes engine_id,
                world_size, kv_connector_extra_config)
            lmcache_engine: Optional LMCacheEngine instance for
                bypass lookup client

        Returns:
            A lookup client instance
        """

        client: LookupClientInterface
        # Check if external_lookup_client is configured
        if config.external_lookup_client is not None:
            if config.enable_async_loading:
                raise ValueError(
                    "Asynchronous loading is not supported for external lookup clients."
                )
            client = LookupClientFactory._create_external_lookup_client(
                config.external_lookup_client, config, metadata
            )
        else:
            # First Party
            from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
                LMCacheAsyncLookupClient,
            )
            from lmcache.v1.lookup_client.lmcache_lookup_client import (
                LMCacheLookupClient,
            )

            # Check if bypass lookup is enabled and lmcache_engine is provided
            if config.enable_scheduler_bypass_lookup and lmcache_engine is not None:
                client = LMCacheBypassLookupClient(config, metadata, lmcache_engine)
            elif config.enable_async_loading:
                client = LMCacheAsyncLookupClient(config, metadata)
            else:
                transport = LookupClientFactory._create_zmq_client_transport(
                    config, metadata
                )
                client = LMCacheLookupClient(config, metadata, transport)

        if config.hit_miss_ratio is not None and 0 <= config.hit_miss_ratio <= 1:
            client = HitLimitLookupClient(client, config)

        # Wrap with ChunkStatisticsLookupClient if enabled
        if config.enable_chunk_statistics:
            client = ChunkStatisticsLookupClient(
                client,
                config,
                metadata,
            )
        return client

    @staticmethod
    def create_lookup_server(
        lmcache_engine: LMCacheEngine,
        metadata: LMCacheMetadata,
    ) -> Optional[Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]]:
        """
        Create a lookup server based on the configuration.

        Args:
            lmcache_engine: The LMCache engine instance
            metadata: The LMCache engine metadata (includes engine_id,
                world_size, kv_connector_extra_config, worker_id)

        Returns:
            A lookup server instance, or None if no server should be created
        """
        config = lmcache_engine.config
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 config is expected for lookup server and client"
        )

        lookup_server_worker_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )

        if config.external_lookup_client is None and (
            len(lookup_server_worker_ids) == 0
            or metadata.worker_id in lookup_server_worker_ids
        ):
            # First Party
            from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
                LMCacheAsyncLookupServer,
            )
            from lmcache.v1.lookup_client.lmcache_lookup_client import (
                LMCacheLookupServer,
            )

            if config.enable_async_loading:
                return LMCacheAsyncLookupServer(lmcache_engine, metadata)
            else:
                transport = LookupClientFactory._create_zmq_server_transport(metadata)
                return LMCacheLookupServer(lmcache_engine, metadata, transport)

        return None

    @staticmethod
    def _create_external_lookup_client(
        external_lookup_uri: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> LookupClientInterface:
        """
        Create an external lookup client based on the URI format.

        Args:
            external_lookup_uri: URI in format <scheme>://<address>
            config: The LMCache engine configuration
            metadata: The LMCache engine metadata

        Returns:
            A lookup client instance

        Raises:
            ValueError: If the URI format is unsupported
        """
        # Parse URI scheme and address
        if "://" not in external_lookup_uri:
            raise ValueError(
                f"Invalid external lookup client URI format: {external_lookup_uri}. "
                "Expected format: <scheme>://<address>"
            )

        scheme, address = external_lookup_uri.split("://", 1)

        # Route to appropriate client based on scheme
        if scheme == "mooncakestore":
            return LookupClientFactory._create_mooncake_lookup_client(
                address, config, metadata
            )
        else:
            raise ValueError(
                f"Unsupported external lookup client scheme: {scheme}. "
                "Supported schemes: mooncakestore"
            )

    @staticmethod
    def _create_mooncake_lookup_client(
        master_address: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> "MooncakeLookupClient":
        """Create a MooncakeLookupClient instance."""
        return MooncakeLookupClient(config, metadata, master_address)

    @staticmethod
    def _create_zmq_client_transport(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> ZmqReqRepClientTransport:
        """Create a ZMQ REQ-REP client transport."""
        kv_extra = metadata.kv_connector_extra_config or {}
        rpc_port = kv_extra.get("lmcache_rpc_port", 0)
        assert metadata.engine_id is not None, (
            "engine_id is required for RPC communication"
        )

        lookup_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        ranks = lookup_ids if len(lookup_ids) > 0 else list(range(metadata.world_size))

        socket_params = [
            SocketParams(
                socket_path=get_zmq_rpc_path_lmcache(
                    metadata.engine_id,
                    "lookup",
                    rpc_port,
                    rank,
                ),
                rank=rank,
            )
            for rank in ranks
        ]
        return ZmqReqRepClientTransport(
            socket_params=socket_params,
            timeout_ms=config.lookup_timeout_ms,
        )

    @staticmethod
    def _create_zmq_server_transport(
        metadata: LMCacheMetadata,
    ) -> ZmqRouterServerTransport:
        """Create a ZMQ ROUTER server transport."""
        kv_extra = metadata.kv_connector_extra_config or {}
        rpc_port = kv_extra.get("lmcache_rpc_port", 0)
        assert metadata.engine_id is not None, (
            "engine_id is required for RPC communication"
        )
        socket_path = get_zmq_rpc_path_lmcache(
            metadata.engine_id,
            "lookup",
            rpc_port,
            metadata.worker_id,
        )
        return ZmqRouterServerTransport(
            socket_path=socket_path,
        )
