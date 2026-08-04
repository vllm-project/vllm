# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import ConnectorAdapter, ConnectorContext
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class BlackholeConnectorAdapter(ConnectorAdapter):
    """Adapter for Blackhole connectors (for testing)."""

    def __init__(self) -> None:
        super().__init__("blackhole://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .blackhole_connector import BlackholeConnector

        logger.info(f"Creating Blackhole connector for URL: {context.url}")
        return BlackholeConnector()
