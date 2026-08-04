# SPDX-License-Identifier: Apache-2.0
# Standard

# Standard
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List, Optional
import asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


@dataclass
class MockMemoryObj:
    metadata: MemoryObjMetadata
    num_bytes: int

    @staticmethod
    def from_tensor_memory_obj(tensor_memory_obj: MemoryObj) -> "MockMemoryObj":
        assert isinstance(tensor_memory_obj, TensorMemoryObj)
        return MockMemoryObj(
            metadata=tensor_memory_obj.metadata,
            num_bytes=len(tensor_memory_obj.byte_array),
        )


class AsyncLRU:
    """
    the async lock protects against race conditions while mimicking synchronization
    being done on a remote server (async client)
    """

    def __init__(self, capacity: int):
        self.lock = asyncio.Lock()
        # current size in bytes
        self.size = 0
        self.capacity = capacity * 1024**3
        self.dict: OrderedDict[CacheEngineKey, MockMemoryObj] = OrderedDict()

    async def exists(self, key: CacheEngineKey):
        async with self.lock:
            if key in self.dict:
                self.dict.move_to_end(key)
                return True
            return False

    async def get(self, key: CacheEngineKey) -> Optional[MockMemoryObj]:
        async with self.lock:
            if key not in self.dict:
                return None
            self.dict.move_to_end(key)
            return self.dict[key]

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MockMemoryObj]]:
        async with self.lock:
            return [self.dict.get(key, None) for key in keys]

    async def put(self, key: CacheEngineKey, mock_obj: MockMemoryObj):
        async with self.lock:
            alloc_size = mock_obj.num_bytes
            if alloc_size > self.capacity:
                raise ValueError(
                    f"Allocation size {alloc_size} is",
                    " greater than capacity {self.capacity}",
                )
            if key in self.dict:
                self.dict.move_to_end(key)
                return None
            self.dict[key] = mock_obj
            while self.size + alloc_size > self.capacity:
                _, mock_obj = self.dict.popitem(last=False)
                self.size -= mock_obj.num_bytes
            self.size += alloc_size

    async def list(self) -> List[CacheEngineKey]:
        async with self.lock:
            return list(self.dict.keys())

    async def close(self):
        async with self.lock:
            self.dict.clear()


class PressureManager:
    """
    Manage I/O pressure of the mock connector
    Assumption: Read and Write throughput are independent
    Locks control overall backend throughput, not per-operation throughput
    """

    def __init__(
        self,
        peeking_latency: float,
        read_throughput: float,
        write_throughput: float,
    ):
        # seconds
        self.peeking_latency = peeking_latency / 1000
        # seconds / byte
        self.read_latency_per_byte = (1 / read_throughput) / 1024**3
        self.write_latency_per_byte = (1 / write_throughput) / 1024**3

        self.read_lock = asyncio.Lock()
        self.write_lock = asyncio.Lock()

    async def on_exists(self):
        # exists latency will delay everyone
        logger.debug(f"waiting {self.peeking_latency} seconds to peek")
        if self.peeking_latency > 0:
            await asyncio.sleep(self.peeking_latency)

    async def on_put(self, mock_obj: MockMemoryObj):
        total_wait_time = self.write_latency_per_byte * mock_obj.num_bytes
        logger.debug(
            f"waiting {total_wait_time} seconds to put {mock_obj.num_bytes} bytes"
        )
        async with self.write_lock:
            await asyncio.sleep(total_wait_time)

    async def on_get(self, mock_obj: MockMemoryObj):
        total_wait_time = self.read_latency_per_byte * mock_obj.num_bytes
        logger.debug(
            f"waiting {total_wait_time} seconds to get {mock_obj.num_bytes} bytes"
        )
        async with self.read_lock:
            await asyncio.sleep(total_wait_time)

    async def on_batched_get(self, mock_objs: List[Optional[MockMemoryObj]]):
        total_bytes = 0
        for mock_obj in mock_objs:
            if mock_obj is None:
                continue
            total_bytes += mock_obj.num_bytes
        total_wait_time = self.read_latency_per_byte * total_bytes
        logger.debug(f"waiting {total_wait_time} seconds to get {total_bytes} bytes")
        async with self.read_lock:
            await asyncio.sleep(total_wait_time)


class MockConnector(RemoteConnector):
    """
    A CPU "remote" backend that doesn't actually go through any network/DB layers and
    let's you manually set R/W throughput and peek latency
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        capacity: int,
        peeking_latency: float = 1.0,
        read_throughput: float = 2.0,
        write_throughput: float = 2.0,
    ):
        """
        peeking_latency: latency for peeking a key (ms)
        capacity: capacity in GB
        read_throughput: GB/s for reading
        write_throughput: GB/s for writing
        """
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.lru_store = AsyncLRU(capacity)

        self.pressure_manager = PressureManager(
            peeking_latency=peeking_latency,
            read_throughput=read_throughput,
            write_throughput=write_throughput,
        )

        # only for the __repr__ string
        self.capacity = capacity
        self.peeking_latency = peeking_latency
        self.read_throughput = read_throughput
        self.write_throughput = write_throughput

        # only for the async loading codepath
        # MockConnector is naturally async so we don't need to use a thread pool
        # for avoiding R-W interference
        self.pq_executor = AsyncPQExecutor(loop)

    async def _exists(self, key: CacheEngineKey) -> bool:
        await self.pressure_manager.on_exists()
        return await self.lru_store.exists(key)

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Synchronous exists check without async lock (for testing purposes)"""
        return key in self.lru_store.dict

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        mock_obj = await self.lru_store.get(key)
        if mock_obj is None:
            return None
        await self.pressure_manager.on_get(mock_obj)
        metadata = mock_obj.metadata
        memory_obj = self.local_cpu_backend.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )

        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None
        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        mock_obj = MockMemoryObj.from_tensor_memory_obj(memory_obj)
        await self.lru_store.put(key, mock_obj)
        await self.pressure_manager.on_put(mock_obj)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    async def list(self) -> List[str]:
        keys = await self.lru_store.list()
        return [k.to_string() for k in keys]

    def support_batched_get(self) -> bool:
        return True

    async def _batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        mock_objs = await self.lru_store.batched_get(keys)
        await self.pressure_manager.on_batched_get(mock_objs)
        memory_objs = []

        for i, mock_obj in enumerate(mock_objs):
            if mock_obj is None:
                logger.warning(
                    f"Mock object is None on {i}",
                    f" out of {len(mock_objs)} objects",
                )
                break
            metadata = mock_obj.metadata
            memory_obj = self.local_cpu_backend.allocate(
                metadata.shape,
                metadata.dtype,
                metadata.fmt,
            )
            if memory_obj is None:
                logger.warning(
                    "Failed to allocate memory even with",
                    f" busy loop on {i} out of {len(mock_objs)} objects",
                )
                break
            memory_objs.append(memory_obj)

        return memory_objs

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        return await self.pq_executor.submit_job(
            self._batched_get, keys=keys, priority=Priorities.GET
        )

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        for key in keys:
            await self.pressure_manager.on_exists()
            if not await self.lru_store.exists(key):
                return num_hit_counts
            num_hit_counts += 1
        return num_hit_counts

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.PEEK,
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # batched get is already async and the non-blocking element is handled
        # in the StorageManager
        return await self.pq_executor.submit_job(
            self._batched_get, keys=keys, priority=Priorities.PREFETCH
        )

    async def close(self):
        await self.lru_store.close()
        await self.pq_executor.shutdown(wait=True)

    def __repr__(self) -> str:
        return (
            f"MockConnector(capacity={self.capacity}GB, "
            f"peeking_latency={self.peeking_latency}ms, "
            f"read_throughput={self.read_throughput}GB/s, "
            f"write_throughput={self.write_throughput}GB/s)"
        )
