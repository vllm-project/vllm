# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    extract_plugin_type,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.hfbucket_connector import (
    PLUGIN_TYPE,
    HFBucketConnector,
    resolve_hfbucket_connector_config,
)

logger = init_logger(__name__)


class HFBucketConnectorAdapter(ConnectorAdapter):
    """Adapter for Hugging Face Buckets remote storage."""

    def __init__(self) -> None:
        super().__init__("plugin://")

    def can_parse(self, url: str) -> bool:
        """Match plugin URLs for the built-in ``hfbucket`` connector type."""
        if not url.startswith(self.schema):
            return False
        plugin_name = url[len(self.schema) :]
        return extract_plugin_type(plugin_name) == PLUGIN_TYPE

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """Create a configured ``HFBucketConnector`` for the given context."""
        if context.config is None:
            raise ValueError("config is required for HFBucketConnector")
        if context.metadata is None:
            raise ValueError("metadata is required for HFBucketConnector")

        plugin_name = context.plugin_name or PLUGIN_TYPE
        connector_config = resolve_hfbucket_connector_config(
            context.config,
            plugin_name=plugin_name,
        )

        logger.info(
            "Creating HFBucket connector for plugin %s and bucket %s",
            plugin_name,
            connector_config.bucket_location.bucket_id,
        )
        return HFBucketConnector(
            local_cpu_backend=context.local_cpu_backend,
            config=context.config,
            metadata=context.metadata,
            connector_config=connector_config,
        )
