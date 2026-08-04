# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class InstrumentedRemoteConnector(RemoteConnector):
    """
    A connector that instruments the underlying connector with
    metrics collection and logging capabilities.
    """

    def __init__(self, connector: RemoteConnector):
        self._connector = connector
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.name = self.__repr__()

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        obj_size = memory_obj.get_size()
        begin = time.perf_counter()
        try:
            await self._connector.put(key, memory_obj)
        finally:
            # Ensure reference count is decreased even if exception occurs
            memory_obj.ref_count_down()

        end = time.perf_counter()
        self._stats_monitor.update_interval_remote_time_to_put((end - begin) * 1000)
        self._stats_monitor.update_interval_remote_write_metrics(obj_size)
        logger.debug(
            "[%s]Bytes offloaded: %.3f MBytes in %.3f ms",
            self.name,
            obj_size / 1e6,
            (end - begin) * 1000,
        )

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        begin = time.perf_counter()
        memory_obj = await self._connector.get(key)
        end = time.perf_counter()
        duration = end - begin

        retrieve_stats = self._stats_monitor.get_current_retrieve_stats()
        if (
            retrieve_stats is not None
            and "remote_backend_individual_get_stats" in retrieve_stats.detailed_metrics
        ):
            retrieve_stats.detailed_metrics["remote_backend_individual_get_stats"][
                key
            ] = {"instrumented_connector_get_time": duration}

        if memory_obj is not None:
            obj_size = memory_obj.get_size()
            self._stats_monitor.update_interval_remote_read_metrics(obj_size)
            logger.debug(
                "[%s]Bytes loaded: %.3f MBytes in %.3f ms",
                self.name,
                obj_size / 1e6,
                duration * 1000,
            )
        return memory_obj

    # Delegate all other methods to the underlying connector
    async def exists(self, key: CacheEngineKey) -> bool:
        return await self._connector.exists(key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return self._connector.exists_sync(key)

    async def list(self) -> List[str]:
        return await self._connector.list()

    async def close(self) -> None:
        await self._connector.close()

    def getWrappedConnector(self) -> RemoteConnector:
        return self._connector

    def support_ping(self) -> bool:
        return self._connector.support_ping()

    async def ping(self) -> int:
        return await self._connector.ping()

    def support_batched_put(self) -> bool:
        return self._connector.support_batched_put()

    def support_batched_get(self) -> bool:
        return self._connector.support_batched_get()

    def support_batched_async_contains(self) -> bool:
        return self._connector.support_batched_async_contains()

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return await self._connector.batched_async_contains(lookup_id, keys, pin)

    def support_batched_get_non_blocking(self) -> bool:
        return self._connector.support_batched_get_non_blocking()

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        begin = time.perf_counter()
        memory_objs = await self._connector.batched_get_non_blocking(lookup_id, keys)
        end = time.perf_counter()
        duration = end - begin

        total_size = sum(
            memory_obj.get_size()
            for memory_obj in memory_objs
            if memory_obj is not None
        )
        if total_size > 0:
            self._stats_monitor.update_interval_remote_read_metrics(total_size)
            logger.debug(
                "[%s]Bytes loaded: %.3f MBytes in %.3f ms",
                self.name,
                total_size / 1e6,
                duration * 1000,
            )
        return memory_objs

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        begin = time.perf_counter()
        memory_objs = await self._connector.batched_get(keys)
        end = time.perf_counter()
        duration = end - begin
        self._stats_monitor.update_interval_remote_time_to_get(duration * 1000)

        retrieve_stats = self._stats_monitor.get_current_retrieve_stats()
        if retrieve_stats is not None:
            retrieve_stats.detailed_metrics[
                "instrumented_connector_batched_get_time"
            ] = (
                retrieve_stats.detailed_metrics.get(
                    "instrumented_connector_batched_get_time", 0.0
                )
                + duration
            )

        total_size = sum(
            memory_obj.get_size()
            for memory_obj in memory_objs
            if memory_obj is not None
        )
        if total_size > 0:
            self._stats_monitor.update_interval_remote_read_metrics(total_size)
            logger.debug(
                "[%s]Bytes loaded: %.3f MBytes in %.3f ms",
                self.name,
                total_size / 1e6,
                duration * 1000,
            )
        return memory_objs

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        total_size = sum(
            memory_obj.get_size()
            for memory_obj in memory_objs
            if memory_obj is not None
        )
        begin = time.perf_counter()
        try:
            await self._connector.batched_put(keys, memory_objs)
        except Exception as e:
            logger.warning(f"batched put error: {e}")
        finally:
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()

        end = time.perf_counter()
        self._stats_monitor.update_interval_remote_time_to_put((end - begin) * 1000)
        self._stats_monitor.update_interval_remote_write_metrics(total_size)
        logger.debug(
            "[%s]Bytes offloaded: %.3f MBytes in %.3f ms",
            self.name,
            total_size / 1e6,
            (end - begin) * 1000,
        )

    def remove_sync(self, key: CacheEngineKey) -> bool:
        return self._connector.remove_sync(key)

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        return self._connector.batched_contains(keys)

    def support_batched_contains(self) -> bool:
        return self._connector.support_batched_contains()

    def reshape_partial_chunk(
        self, memory_obj: MemoryObj, bytes_read: int
    ) -> MemoryObj:
        return self._connector.reshape_partial_chunk(memory_obj, bytes_read)

    def post_init(self):
        return self._connector.post_init()

    def __repr__(self) -> str:
        return f"InstrumentedRemoteConnector({self._connector})"
