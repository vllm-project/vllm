# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import asyncio

# Local
from .connector_client_base import ConnectorClientBase

try:
    # First Party
    from lmcache.lmcache_redis import LMCacheRedisClient

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    LMCacheRedisClient = None  # type: ignore


class RESPClient(ConnectorClientBase[LMCacheRedisClient]):
    def __init__(
        self,
        host: str,
        port: int,
        num_workers: int,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        username: str = "",
        password: str = "",
    ):
        if not REDIS_AVAILABLE:
            raise RuntimeError(
                "RESPClient requires the C++ Redis extension. "
                "Build with: pip install -e ."
            )
        native_client = LMCacheRedisClient(host, port, num_workers, username, password)
        super().__init__(native_client, loop)
