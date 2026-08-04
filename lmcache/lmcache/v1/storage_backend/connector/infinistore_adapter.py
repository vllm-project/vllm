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


class InfinistoreConnectorAdapter(ConnectorAdapter):
    """Adapter for Infinistore connectors."""

    def __init__(self) -> None:
        super().__init__("infinistore://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Third Party
        import infinistore

        # Local
        from .infinistore_connector import InfinistoreConnector

        logger.info(f"Creating Infinistore connector for URL: {context.url}")
        hosts = context.url.split(",")
        if len(hosts) > 1:
            raise ValueError(
                f"Only one host is supported for infinistore, but got {hosts}"
            )

        parse_url = parse_remote_url(context.url)
        device_name = parse_url.query_params.get("device", ["mlx5_0"])[0]

        link_type_str = "LINK_ETHERNET"
        if context.config and context.config.extra_config:
            link_type_str = context.config.extra_config.get(
                "infinistore_link_type", link_type_str
            )

        link_type_str = link_type_str.upper()
        try:
            link_type = getattr(infinistore, link_type_str)
        except AttributeError as e:
            raise ValueError(f"Invalid link_type: {link_type_str}") from e

        return InfinistoreConnector(
            host=parse_url.host,
            port=parse_url.port,
            dev_name=device_name,
            link_type=link_type,
            loop=context.loop,
            memory_allocator=context.local_cpu_backend,
        )
