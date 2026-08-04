# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, List, Tuple
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class RESPConnectorAdapter(ConnectorAdapter):
    """Adapter for RESP connectors."""

    def __init__(self) -> None:
        super().__init__("resp://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .redis_connector import RESPConnector

        config = context.config
        assert config is not None

        # Get config from extra_config with defaults
        extra_config = config.extra_config if config.extra_config is not None else {}

        # Validate that save_chunk_meta and save_unfull_chunk are False for RESP
        self.save_chunk_meta = bool(extra_config.get("save_chunk_meta", False))
        assert not self.save_chunk_meta, "save_chunk_meta must be False for RESP"

        assert not config.save_unfull_chunk, "save_unfull_chunk must be False for RESP"

        # Get number of threads for RESP connection pool (default is 8)
        self.resp_num_threads = int(extra_config.get("resp_num_threads", 8))

        # Config/CLI args take precedence over environment variables,
        # which serve as defaults. This keeps secrets out of logged
        # config while allowing explicit overrides.
        cfg_username = str(extra_config.get("username", ""))
        cfg_password = str(extra_config.get("password", ""))
        username = cfg_username or os.environ.get("LMCACHE_RESP_USERNAME", "")
        password = cfg_password or os.environ.get("LMCACHE_RESP_PASSWORD", "")

        parsed_url = parse_remote_url(context.url)

        # Config/URL values take precedence; env vars are fallback
        host = parsed_url.host or os.environ.get("LMCACHE_RESP_HOST", "")
        port = (
            parsed_url.port
            if parsed_url.port
            else int(os.environ.get("LMCACHE_RESP_PORT", "0"))
        )

        logger.info("Creating RESP connector for %s:%d", host, port)
        return RESPConnector(
            host=host,
            port=port,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            num_threads=self.resp_num_threads,
            username=username,
            password=password,
        )


class RedisConnectorAdapter(ConnectorAdapter):
    """Adapter for Redis connectors."""

    def __init__(self) -> None:
        super().__init__("redis://")

    def can_parse(self, url: str) -> bool:
        return url.startswith((self.schema, "rediss://", "unix://", "plugin://redis"))

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .redis_connector import RedisConnector

        url = context.url
        if url.startswith("plugin://redis"):
            extra_config: Dict[str, Any] = {}
            remote_url = None
            if context.config is not None:
                extra_config = (
                    context.config.extra_config
                    if context.config.extra_config is not None
                    else {}
                )
                remote_url = context.config.remote_url
            cfg_redis_url = extra_config.get(
                "remote_storage_plugin.redis.redis_url"
            ) or extra_config.get("redis_url")
            url = cfg_redis_url or remote_url or "redis://localhost:6379"

        logger.info(f"Creating Redis connector for URL: {url}")
        return RedisConnector(
            url=url,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )


class RedisSentinelConnectorAdapter(ConnectorAdapter):
    """Adapter for Redis Sentinel connectors."""

    def __init__(self) -> None:
        super().__init__("redis-sentinel://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .redis_connector import RedisSentinelConnector

        logger.info(f"Creating Redis Sentinel connector for URL: {context.url}")
        url = context.url[len(self.schema) :]

        # Parse username and password
        username: str = ""
        password: str = ""
        if "@" in url:
            auth, url = url.split("@", 1)
            if ":" in auth:
                username, password = auth.split(":", 1)
            else:
                username = auth

        # Parse host and port
        hosts_and_ports: List[Tuple[str, int]] = []
        assert self.schema is not None
        for sub_url in url.split(","):
            if not sub_url.startswith(self.schema):
                sub_url = self.schema + sub_url

            parsed_url = parse_remote_url(sub_url)
            hosts_and_ports.append((parsed_url.host, parsed_url.port))

        return RedisSentinelConnector(
            hosts_and_ports=hosts_and_ports,
            username=username,
            password=password,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )


class RedisClusterConnectorAdapter(ConnectorAdapter):
    """Adapter for Redis Cluster connectors."""

    def __init__(self) -> None:
        super().__init__("redis-cluster://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .redis_connector import RedisClusterConnector

        logger.info(f"Creating Redis Cluster connector for URL: {context.url}")
        url = context.url[len(self.schema) :]

        # Parse username and password
        username: str = ""
        password: str = ""
        if "@" in url:
            auth, url = url.split("@", 1)
            if ":" in auth:
                username, password = auth.split(":", 1)
            else:
                username = auth

        # Parse host and port
        hosts_and_ports: List[Tuple[str, int]] = []
        assert self.schema is not None
        for sub_url in url.split(","):
            if not sub_url.startswith(self.schema):
                sub_url = self.schema + sub_url

            parsed_url = parse_remote_url(sub_url)
            hosts_and_ports.append((parsed_url.host, parsed_url.port))

        return RedisClusterConnector(
            hosts_and_ports=hosts_and_ports,
            username=username,
            password=password,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )
