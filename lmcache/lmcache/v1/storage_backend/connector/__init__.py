# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse
import asyncio
import importlib
import inspect

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.utils.subclass_discovery import discover_subclasses

logger = init_logger(__name__)


@dataclass
class ParsedRemoteURL:
    """
    The parsed URL of the format:
        <host>:<port>[/path][?query]
    """

    host: str
    port: int
    path: str
    username: Optional[str] = None
    password: Optional[str] = None
    query_params: Dict[str, List[str]] = field(default_factory=dict)


def parse_remote_url(url: str) -> ParsedRemoteURL:
    """
    Parses the remote URL into its constituent parts with support for:
    - Multiple hosts (comma-separated)
    - Path and query parameters in each host definition
    - Forward compatibility with legacy format

    Args:
        url: The URL to parse

    Returns:
        ParsedRemoteURL: The parsed URL components

    Raises:
        ValueError: If the URL is invalid
    """

    logger.debug(f"Parsing remote URL: {url}")
    parsed = urlparse(url)

    username = parsed.username
    password = parsed.password
    host = parsed.hostname
    port = parsed.port
    path = parsed.path if parsed.path else ""
    query = parse_qs(parsed.query) if parsed.query else {}

    assert host is not None, f"Invalid URL {url}: missing host"
    assert port is not None, f"Invalid URL {url}: missing port"
    return ParsedRemoteURL(
        host=host,
        port=port,
        path=path,
        username=username,
        password=password,
        query_params=query,
    )


class SafeLocalCPUBackend(LocalCPUBackend):
    """
    A safe stub for LocalCPUBackend that can be used when local_cpu_backend is None.
    """

    def __init__(self, config: LMCacheEngineConfig):
        pass

    def allocate(self, *args, **kwargs):
        raise RuntimeError(
            "SafeLocalCPUBackend.allocate() should never be called. "
            "This indicates a bug where scheduler role is trying to allocate memory."
        )

    def __str__(self):
        return "SafeLocalCPUBackend(dummy)"


def extract_plugin_type(plugin_name: str) -> str:
    """Extract the type portion from a plugin name.

    Plugin name format: ``{type}`` or ``{type}.{instance}``.
    Returns the *type* part so that adapters can match by type.

    Examples:
        >>> extract_plugin_type("fs")
        'fs'
        >>> extract_plugin_type("fs.primary")
        'fs'
    """
    return plugin_name.split(".", 1)[0]


class ConnectorContext:
    """
    Context for creating a connector.

    Attributes:
        url: The remote URL
        loop: The asyncio event loop
        local_cpu_backend: The local CPU backend
            (wrapped as SafeLocalCPUBackend if None)
        config: Optional LMCache engine configuration
        plugin_name: Optional plugin instance name
            (e.g. "fs", "fs.primary")
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional[LocalCPUBackend],
        config: Optional[LMCacheEngineConfig],
        metadata: Optional[LMCacheMetadata],
        plugin_name: Optional[str] = None,
    ):
        self.url = url
        self.loop = loop
        # Wrap None as SafeLocalCPUBackend to satisfy type requirements
        # The SafeLocalCPUBackend will raise an error if allocate() is called
        self.local_cpu_backend: LocalCPUBackend = (
            local_cpu_backend
            if local_cpu_backend is not None
            else SafeLocalCPUBackend(config)
        )
        self.config = config
        self.metadata = metadata
        self.plugin_name = plugin_name

    def get_full_chunk_size_bytes(self) -> int:
        """
        return the number of bytes in a full chunk
        useful for S3Connector where we need to preallocate filesystem buffers
        in ramfs for zero-copy transfers
        """
        return self.local_cpu_backend.get_full_chunk_size_bytes()


class ConnectorAdapter(ABC):
    """Base class for connector adapters."""

    def __init__(self, schema: str = "") -> None:
        self.schema = schema

    def can_parse(self, url: str) -> bool:
        return self.schema != "" and url.startswith(self.schema)

    @abstractmethod
    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create a connector using the given context.
        """
        pass


class DynamicConnectorAdapter(ConnectorAdapter):
    """Adapter that wraps a RemoteConnector class loaded
    dynamically from plugin config.

    When ``class_name`` points to a ``RemoteConnector`` subclass
    rather than a ``ConnectorAdapter``, this wrapper is used to
    instantiate the connector with the proper context.
    """

    def __init__(
        self,
        plugin_name: str,
        connector_class: type,
    ) -> None:
        schema = "plugin://%s" % extract_plugin_type(plugin_name)
        super().__init__(schema)
        self._plugin_name = plugin_name
        self._connector_class = connector_class

    def can_parse(self, url: str) -> bool:
        if url.startswith(self.schema):
            return True
        if url.startswith("plugin://"):
            pname = url[len("plugin://") :]
            return extract_plugin_type(pname) == extract_plugin_type(self._plugin_name)
        return False

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        logger.info(
            "Creating dynamic connector %s via %s",
            self._plugin_name,
            self._connector_class.__name__,
        )
        return self._connector_class(
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            config=context.config,
        )


class ConnectorManager:
    """
    Manager for creating connectors based on URL.

    This class maintains a registry of connector adapters and creates
    the appropriate connector based on the URL.
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional[LocalCPUBackend],
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheMetadata] = None,
        plugin_name: Optional[str] = None,
    ) -> None:
        logger.info("Initializing ConnectorManager")
        self.context = ConnectorContext(
            url=url,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            config=config,
            metadata=metadata,
            plugin_name=plugin_name,
        )
        self.adapters: List[ConnectorAdapter] = []
        self._remote_adapters_builtin_launcher()
        self._remote_adapters_plugin_launcher(config)

    def _remote_adapters_builtin_launcher(self) -> None:
        """Automatically load all builtin remote connector adapters."""
        for cls in discover_subclasses(
            "lmcache.v1.storage_backend.connector",
            ConnectorAdapter,  # type: ignore[type-abstract]
            module_filter=lambda name: (
                not name.startswith("_") and name.endswith("_adapter")
            ),
            require_defined_in_module=False,
        ):
            try:
                self.adapters.append(cls())
                logger.info(f"Discovered adapter: {cls.__name__}")
            except Exception as e:
                logger.error(f"Failed to instantiate adapter {cls.__name__}: {str(e)}")

    def _remote_adapters_plugin_launcher(self, config: LMCacheEngineConfig) -> None:
        """Automatically load all plug and play remote connector adapters."""

        if config is None:
            logger.warning(
                "Configuration not available to parse remote connector adapters."
            )
            return

        # Get the list of allowed remote connector adapters if configured
        remote_storage_plugins = (
            set(config.remote_storage_plugins)
            if config.remote_storage_plugins
            else set()
        )

        for remote_storage_plugin in remote_storage_plugins:
            try:
                extra_config = config.extra_config

                module_path = (
                    extra_config.get(
                        "remote_storage_plugin.%s.module_path" % remote_storage_plugin
                    )
                    if extra_config
                    else None
                )
                class_name = (
                    extra_config.get(
                        "remote_storage_plugin.%s.class_name" % remote_storage_plugin
                    )
                    if extra_config
                    else None
                )

                if not module_path or not class_name:
                    # Skip silently when a builtin adapter
                    # already handles this plugin type.
                    plugin_url = "plugin://%s" % remote_storage_plugin
                    if any(a.can_parse(plugin_url) for a in self.adapters):
                        continue
                    logger.warning(
                        "Remote connector %s missing adapter module_path or class_name",
                        remote_storage_plugin,
                    )
                    continue

                # Dynamically import the module
                module = importlib.import_module(module_path)
                # Get the class from the module
                loaded_class = getattr(module, class_name)

                if inspect.isclass(loaded_class) and issubclass(
                    loaded_class, ConnectorAdapter
                ):
                    adapter_instance = loaded_class()
                elif inspect.isclass(loaded_class) and issubclass(
                    loaded_class, RemoteConnector
                ):
                    adapter_instance = DynamicConnectorAdapter(
                        plugin_name=remote_storage_plugin,
                        connector_class=loaded_class,
                    )
                else:
                    logger.warning(
                        "Remote connector %s class %s is "
                        "neither a ConnectorAdapter nor a "
                        "RemoteConnector subclass",
                        remote_storage_plugin,
                        class_name,
                    )
                    continue
                self.adapters.append(adapter_instance)
                logger.info(
                    "Discovered adapter: %s",
                    loaded_class.__name__,
                )
            except (ImportError, AttributeError) as e:
                logger.error(
                    f"Failed to load remote connector {remote_storage_plugin} due to "
                    f"import/attribute error: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to create remote connector {remote_storage_plugin} "
                    f"adapter: {str(e)}"
                )

    def create_connector(self) -> RemoteConnector:
        for adapter in self.adapters:
            if adapter.can_parse(self.context.url):
                logger.info(f"Creating connector for URL: {self.context.url}")
                connector = adapter.create_connector(self.context)
                return connector

        raise ValueError(f"No adapter found for URL: {self.context.url}")


def CreateConnector(
    url: str,
    loop: asyncio.AbstractEventLoop,
    local_cpu_backend: Optional[LocalCPUBackend],
    config: Optional[LMCacheEngineConfig] = None,
    metadata: Optional[LMCacheMetadata] = None,
    plugin_name: Optional[str] = None,
) -> InstrumentedRemoteConnector:
    """
    Create a remote connector from the given URL.

    Supported URL formats:
    - redis://[[username]:[password]@]host[:port][/database][?option=value]
    - rediss://[[username]:[password]@]host[:port][/database][?option=value] (SSL)
    - redis-sentinel://[[username]:[password]@]host1:port1[,host2:port2,...]/service_name
    - lm://host:port
    - infinistore://host:port[?device=device_name]
    - mooncakestore://host:port[?device=device_name]
    - blackhole://[any_text]
    - audit://host:port[?verify=true|false]
    - fs://[host:port]/path
    - s3://[bucket].s3express-[az_id].[region].amazonaws.com"
    - mock://[capacity]/?peeking_latency=[ms]&read_throughput=[GB/s]&write_throughput=[GB/s]
    or
    - s3://[bucket].s3.[region].amazonaws.com

    Examples:
    - redis://localhost:6379
    - rediss://user:password@redis.example.com:6380/0
    - redis-sentinel://user:password@sentinel1:26379,sentinel2:26379/mymaster
    - lm://localhost:65432
    - infinistore://127.0.0.1:12345?device=mlx5_0
    - mooncakestore://127.0.0.1:50051
    - blackhole://
    - audit://localhost:8080?verify=true
    - fs:///tmp/lmcache
    - external://host:0/external_log_connector.lmc_external_log_connector/?connector_name=ExternalLogConnector
    - s3://fakefile--use1-az4--x-s3.s3express-use1-az4.us-east-1.amazonaws.com
    - mock://100/?peeking_latency=1&read_throughput=2&write_throughput=2
    or
    - s3://fakefile--use1-az4--x-s3.s3.us-east-1.amazonaws.com

    Args:
        url: The remote URL
        loop: The asyncio event loop
        local_cpu_backend: The local CPU backend (can be None for scheduler role)
        config: Optional LMCache engine configuration
        metadata: Optional LMCache engine metadata

    Returns:
        RemoteConnector: The created connector

    Raises:
        ValueError: If the connector cannot be created
    """

    # Basic URL validation - check for scheme
    if "://" not in url:
        raise ValueError(f"Invalid remote url {url}: missing scheme")

    manager = ConnectorManager(
        url, loop, local_cpu_backend, config, metadata, plugin_name
    )
    connector = manager.create_connector()

    return InstrumentedRemoteConnector(connector)
