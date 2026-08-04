# SPDX-License-Identifier: Apache-2.0
"""
Aerospike L2 adapter config and factory.

Backed by the native C++ Aerospike connector wrapped with
``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Optional
import os

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory

logger = init_logger(__name__)


class AerospikeL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native Aerospike connector.

    Fields:
    - hosts: seed hosts as ``host:port[,host:port...]``
    - namespace: Aerospike namespace
    - set_name: Aerospike set name
    - num_workers: C++ worker threads for I/O (default 8)
    - read_timeout_ms / write_timeout_ms: client timeouts
    - default_ttl_seconds: record TTL (0 = namespace default)
    - target_segment_bytes: shard target (0 = use discovered cap)
    - max_record_bytes: override server record cap (0 = discover)
    - username / password: optional EE auth
    - max_capacity_gb: L2 capacity tracking (0 = disabled)
    """

    def __init__(
        self,
        hosts: str,
        namespace: str = "lmcache",
        set_name: str = "kv_chunks",
        num_workers: int = 8,
        read_timeout_ms: int = 1000,
        write_timeout_ms: int = 2000,
        default_ttl_seconds: int = 86400,
        target_segment_bytes: int = 0,
        max_record_bytes: int = 0,
        username: str = "",
        password: str = "",
        max_capacity_gb: float = 0,
    ) -> None:
        super().__init__()
        self.hosts = hosts
        self.namespace = namespace
        self.set_name = set_name
        self.num_workers = num_workers
        self.read_timeout_ms = read_timeout_ms
        self.write_timeout_ms = write_timeout_ms
        self.default_ttl_seconds = default_ttl_seconds
        self.target_segment_bytes = target_segment_bytes
        self.max_record_bytes = max_record_bytes
        self.username = username
        self.password = password
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "AerospikeL2AdapterConfig":
        """Construct a config from a raw configuration dict.

        Args:
            d: Raw configuration dict (typically from JSON/CLI).

        Returns:
            A validated ``AerospikeL2AdapterConfig``.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        hosts = d.get("hosts")
        if not isinstance(hosts, str) or not hosts:
            raise ValueError("hosts must be a non-empty string")

        namespace = d.get("namespace", "lmcache")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be a non-empty string")

        set_name = d.get("set_name", d.get("set", "kv_chunks"))
        if not isinstance(set_name, str) or not set_name:
            raise ValueError("set_name must be a non-empty string")

        num_workers = d.get("num_workers", 8)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        read_timeout_ms = d.get("read_timeout_ms", 1000)
        if not isinstance(read_timeout_ms, int) or read_timeout_ms <= 0:
            raise ValueError("read_timeout_ms must be a positive integer")

        write_timeout_ms = d.get("write_timeout_ms", 2000)
        if not isinstance(write_timeout_ms, int) or write_timeout_ms <= 0:
            raise ValueError("write_timeout_ms must be a positive integer")

        default_ttl_seconds = d.get("default_ttl_seconds", 86400)
        if not isinstance(default_ttl_seconds, int) or default_ttl_seconds < 0:
            raise ValueError("default_ttl_seconds must be a non-negative integer")

        target_segment_bytes = d.get("target_segment_bytes", 0)
        if not isinstance(target_segment_bytes, int) or target_segment_bytes < 0:
            raise ValueError("target_segment_bytes must be a non-negative integer")

        max_record_bytes = d.get("max_record_bytes", 0)
        if not isinstance(max_record_bytes, int) or max_record_bytes < 0:
            raise ValueError("max_record_bytes must be a non-negative integer")

        username = d.get("username", "")
        password = d.get("password", "")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")

        return cls(
            hosts=hosts,
            namespace=str(namespace),
            set_name=str(set_name),
            num_workers=num_workers,
            read_timeout_ms=read_timeout_ms,
            write_timeout_ms=write_timeout_ms,
            default_ttl_seconds=default_ttl_seconds,
            target_segment_bytes=target_segment_bytes,
            max_record_bytes=max_record_bytes,
            username=str(username),
            password=str(password),
            max_capacity_gb=float(max_capacity_gb),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Aerospike L2 adapter config fields:\n"
            "- hosts (str): seed hosts host:port[,...] (required)\n"
            "- namespace (str): Aerospike namespace (default lmcache)\n"
            "- set_name / set (str): Aerospike set (default kv_chunks)\n"
            "- num_workers (int): C++ I/O threads (default 8)\n"
            "- read_timeout_ms (int): read timeout ms (default 1000)\n"
            "- write_timeout_ms (int): write timeout ms (default 2000)\n"
            "- default_ttl_seconds (int): record TTL (default 86400)\n"
            "- target_segment_bytes (int): shard target, 0 = auto (default 0)\n"
            "- max_record_bytes (int): cap override, 0 = discover (default 0)\n"
            "- username / password (str): optional auth\n"
            "- max_capacity_gb (float): L2 capacity for eviction (default 0)\n\n"
            "Environment variable defaults (when config value is empty):\n"
            "- LMCACHE_AEROSPIKE_HOSTS\n"
            "- LMCACHE_AEROSPIKE_NAMESPACE\n"
            "- LMCACHE_AEROSPIKE_SET\n"
            "- LMCACHE_AEROSPIKE_USERNAME\n"
            "- LMCACHE_AEROSPIKE_PASSWORD"
        )


def _create_aerospike_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the C++ Aerospike connector."""
    del l1_memory_desc
    try:
        # First Party
        from lmcache.lmcache_aerospike import LMCacheAerospikeClient
    except ImportError as e:
        raise RuntimeError(
            "Aerospike L2 adapter requires the C++ Aerospike extension. "
            "Build with: BUILD_AEROSPIKE=1 pip install -e ."
        ) from e

    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, AerospikeL2AdapterConfig)

    hosts = config.hosts or os.environ.get("LMCACHE_AEROSPIKE_HOSTS", "")
    namespace = config.namespace or os.environ.get(
        "LMCACHE_AEROSPIKE_NAMESPACE", "lmcache"
    )
    set_name = config.set_name or os.environ.get("LMCACHE_AEROSPIKE_SET", "kv_chunks")
    username = config.username or os.environ.get("LMCACHE_AEROSPIKE_USERNAME", "")
    password = config.password or os.environ.get("LMCACHE_AEROSPIKE_PASSWORD", "")

    if not hosts:
        raise ValueError("hosts must be a non-empty string")

    native_client = LMCacheAerospikeClient(
        hosts,
        namespace,
        set_name,
        config.num_workers,
        config.read_timeout_ms,
        config.write_timeout_ms,
        config.default_ttl_seconds,
        config.target_segment_bytes,
        config.max_record_bytes,
        username,
        password,
    )
    logger.info(
        "Created Aerospike L2 adapter: hosts=%s namespace=%s set=%s (workers=%d)",
        hosts,
        namespace,
        set_name,
        config.num_workers,
    )
    return NativeConnectorL2Adapter(
        native_client,
        max_capacity_gb=config.max_capacity_gb,
        type_name="aerospike",
        extra_status={
            "hosts": hosts,
            "namespace": namespace,
            "set_name": set_name,
            "num_workers": config.num_workers,
        },
    )


register_l2_adapter_type("aerospike", AerospikeL2AdapterConfig)
register_l2_adapter_factory("aerospike", _create_aerospike_l2_adapter)
