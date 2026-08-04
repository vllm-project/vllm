# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    extract_plugin_type,
)
from lmcache.v1.storage_backend.connector.base_connector import (
    RemoteConnector,
)

logger = init_logger(__name__)

PLUGIN_TYPE = "mooncakestore"


class MooncakestoreConnectorAdapter(ConnectorAdapter):
    """Adapter for Mooncakestore connectors."""

    def __init__(self) -> None:
        super().__init__("mooncakestore://")

    def can_parse(self, url: str) -> bool:
        if url.startswith(self.schema):
            return True
        if url.startswith("plugin://"):
            pname = url[len("plugin://") :]
            return extract_plugin_type(pname) == PLUGIN_TYPE
        return False

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .mooncakestore_connector import MooncakestoreConnector

        logger.info("Creating Mooncakestore connector")

        return MooncakestoreConnector(
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            lmcache_config=context.config,
            plugin_name=context.plugin_name,
        )
