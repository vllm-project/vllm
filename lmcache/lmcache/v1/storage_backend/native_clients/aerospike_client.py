# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import asyncio

# Local
from .connector_client_base import ConnectorClientBase

try:
    # First Party
    from lmcache.lmcache_aerospike import LMCacheAerospikeClient

    AEROSPIKE_AVAILABLE = True
except ImportError:
    AEROSPIKE_AVAILABLE = False
    LMCacheAerospikeClient = None  # type: ignore


class AerospikeClient(ConnectorClientBase["LMCacheAerospikeClient"]):
    """Non-MP remote backend client wrapping ``LMCacheAerospikeClient``."""

    def __init__(
        self,
        hosts: str,
        namespace: str,
        set_name: str,
        num_workers: int,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        read_timeout_ms: int = 1000,
        write_timeout_ms: int = 2000,
        default_ttl_seconds: int = 86400,
        target_segment_bytes: int = 0,
        max_record_bytes: int = 0,
        username: str = "",
        password: str = "",
    ) -> None:
        """Create a client backed by the native Aerospike connector.

        Args:
            hosts: Seed hosts as ``host:port[,host:port...]``.
            namespace: Aerospike namespace.
            set_name: Aerospike set name.
            num_workers: C++ worker threads for I/O.
            loop: Optional asyncio loop for async wrappers.
            read_timeout_ms: Read timeout in milliseconds.
            write_timeout_ms: Write timeout in milliseconds.
            default_ttl_seconds: Record TTL (0 = namespace default).
            target_segment_bytes: Shard target (0 = use discovered cap).
            max_record_bytes: Override server record cap (0 = discover).
            username: Optional auth username.
            password: Optional auth password.

        Raises:
            RuntimeError: If the C++ Aerospike extension is not built.
        """
        if not AEROSPIKE_AVAILABLE:
            raise RuntimeError(
                "AerospikeClient requires the C++ Aerospike extension. "
                "Build with: BUILD_AEROSPIKE=1 pip install -e ."
            )
        native_client = LMCacheAerospikeClient(
            hosts,
            namespace,
            set_name,
            num_workers,
            read_timeout_ms,
            write_timeout_ms,
            default_ttl_seconds,
            target_segment_bytes,
            max_record_bytes,
            username,
            password,
        )
        super().__init__(native_client, loop)
