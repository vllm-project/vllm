# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, no_type_check

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj

# reuse
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class BlackholeConnector(RemoteConnector):
    def __init__(self):
        pass

    async def exists(self, key: CacheEngineKey) -> bool:
        return False

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return False

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        pass

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        logger.info("Closed the blackhole connection")
