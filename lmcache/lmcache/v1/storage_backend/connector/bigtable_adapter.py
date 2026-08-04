# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.v1.storage_backend.connector import ConnectorAdapter, ConnectorContext
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

# Local
from .bigtable_connector import BigtableConnector


class BigtableConnectorAdapter(ConnectorAdapter):
    """Adapter for BigtableConnector to integrate natively with LMCache
    built-in connectors.

    Supports both 'plugin://bigtable' and 'bigtable://' URL schemas.
    """

    def __init__(self):
        super().__init__("plugin://bigtable")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema) or url.startswith("bigtable://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        return BigtableConnector(
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            config=context.config,
            plugin_name=context.plugin_name,
        )
