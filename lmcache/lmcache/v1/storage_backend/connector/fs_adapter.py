# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    extract_plugin_type,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import (
    RemoteConnector,
)

logger = init_logger(__name__)

PLUGIN_TYPE = "fs"


class FsConnectorAdapter(ConnectorAdapter):
    """Adapter for Filesystem connectors."""

    def __init__(self) -> None:
        super().__init__("fs://")

    def can_parse(self, url: str) -> bool:
        if url.startswith(self.schema):
            return True
        if url.startswith("plugin://"):
            pname = url[len("plugin://") :]
            return extract_plugin_type(pname) == PLUGIN_TYPE
        return False

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .fs_connector import FSConnector

        logger.info("Creating FS connector")

        # Legacy URL mode: extract base_path from URL
        base_paths_str = None
        if context.plugin_name is None:
            parsed = parse_remote_url(context.url)
            base_paths_str = parsed.path

        return FSConnector(
            context.loop,
            context.local_cpu_backend,
            context.config,
            plugin_name=context.plugin_name,
            base_paths_str=base_paths_str,
        )
