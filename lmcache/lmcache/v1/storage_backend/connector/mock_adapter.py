# SPDX-License-Identifier: Apache-2.0
# Standard
from urllib.parse import parse_qs, urlparse

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class MockConnectorAdapter(ConnectorAdapter):
    """Adapter for Mock Connector"""

    def __init__(self) -> None:
        super().__init__("mock://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local import to avoid circular dependencies
        # Local
        from .mock_connector import MockConnector

        logger.info(f"Creating Mock connector for URL: {context.url}")

        parsed = urlparse(context.url)
        # capacity is provided as the netloc in URLs like: mock://100/?...
        if not parsed.netloc:
            raise ValueError(
                "mock connector requires capacity in GB as netloc, e.g. mock://100/?..."
            )
        try:
            capacity_gb = int(parsed.netloc)
        except ValueError as e:
            raise ValueError(
                f"Invalid capacity '{parsed.netloc}' for",
                " mock connector; must be an integer (GB).",
            ) from e

        params = parse_qs(parsed.query) if parsed.query else {}
        # Defaults
        peeking_latency_ms = float(params.get("peeking_latency", ["1"])[0])
        read_throughput_gbps = float(params.get("read_throughput", ["2"])[0])
        write_throughput_gbps = float(params.get("write_throughput", ["2"])[0])

        return MockConnector(
            url=context.url,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            capacity=capacity_gb,
            read_throughput=read_throughput_gbps,
            write_throughput=write_throughput_gbps,
            peeking_latency=peeking_latency_ms,
        )
