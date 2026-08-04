# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class LMServerConnectorAdapter(ConnectorAdapter):
    """Adapter for LM Server connectors."""

    def __init__(self) -> None:
        super().__init__("lm://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .lm_connector import LMCServerConnector

        logger.info(f"Creating LM Server connector for URL: {context.url}")
        parse_url = parse_remote_url(context.url)
        return LMCServerConnector(
            host=parse_url.host,
            port=parse_url.port,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )
