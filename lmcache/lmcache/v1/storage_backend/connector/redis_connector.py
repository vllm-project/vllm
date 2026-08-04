# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import IntEnum, auto
from typing import List, Optional, Tuple, no_type_check
import asyncio
import inspect
import os

# Third Party
from redis.asyncio.cluster import ClusterNode, RedisCluster
import redis.asyncio as redis

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.native_clients.resp_client import RESPClient

logger = init_logger(__name__)

# TODO(Jiayi): Use `redis.asyncio`
# NOTE(Jiayi): `redis-py` supports async operations, but data copy
# cannot be avoided. `hiredis` is more lower-level but asyncio is
# not supported.


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


class RESPConnector(RemoteConnector):
    """
    The remote url should start with "resp://" and only have one host-port pair
    """

    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        num_threads: int = 8,
        username: str = "",
        password: str = "",
    ):
        # this gives us access to self.full_chunk_size_bytes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        self.host = host
        self.port = port
        self.num_threads = num_threads
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.client = RESPClient(host, port, num_threads, loop, username, password)
        self.pq_executor = AsyncPQExecutor(loop)

    async def _exists(self, key: CacheEngineKey) -> bool:
        return await self.client.exists(key.to_string())

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return self.client.exists_sync(key.to_string())

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes,
            self.meta_dtypes,
            self.meta_fmt,
        )

        # byte array view of tensor
        recv_buf = memory_obj.byte_array
        if not isinstance(recv_buf, memoryview):
            recv_buf = memoryview(recv_buf)
        await self.client.get(key_str, recv_buf)
        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()
        send_buf = memory_obj.byte_array
        if not isinstance(send_buf, memoryview):
            send_buf = memoryview(send_buf)
        await self.client.set(key_str, send_buf)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    def support_batched_put(self) -> bool:
        return True

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        key_strs = [key.to_string() for key in keys]
        send_bufs = [
            memory_obj.byte_array
            if isinstance(memory_obj.byte_array, memoryview)
            else memoryview(memory_obj.byte_array)
            for memory_obj in memory_objs
        ]
        await self.client.batch_set(key_strs, send_bufs)

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        await self.pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    def support_batched_get(self) -> bool:
        return True

    async def _batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        key_strs = [key.to_string() for key in keys]
        memory_objs = [
            self.local_cpu_backend.allocate(
                self.meta_shapes,
                self.meta_dtypes,
                self.meta_fmt,
            )
            for _ in keys
        ]
        recv_bufs = [
            memory_obj.byte_array
            if isinstance(memory_obj.byte_array, memoryview)
            else memoryview(memory_obj.byte_array)
            for memory_obj in memory_objs
        ]
        await self.client.batch_get(key_strs, recv_bufs)
        return memory_objs

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        return await self.pq_executor.submit_job(
            self._batched_get, keys=keys, priority=Priorities.GET
        )

    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        """Synchronous batched contains - checks consecutive prefix existence."""
        key_strs = [key.to_string() for key in keys]
        results = self.client.batch_exists_sync(key_strs)
        count = 0
        # we only want the prefixes
        for result in results:
            if not result:
                return count
            count += 1
        return count

    async def _batched_async_contains(self, keys: List[CacheEngineKey]) -> int:
        key_strs = [key.to_string() for key in keys]
        results = await self.client.batch_exists(key_strs)
        count = 0
        # we only want the prefixes
        for result in results:
            if not result:
                return count
            count += 1
        return count

    def support_batched_async_contains(self) -> bool:
        return True

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Check how many keys exist in file system in batch

        Args:
            lookup_id: Identifier for this lookup operation
            keys: List of keys to check
            pin: Whether to pin the keys (not used in FS connector)

        Returns:
            Number of consecutive keys that exist, starting from the first key
        """
        # prefetch priority
        return await self.pq_executor.submit_job(
            self._batched_async_contains, keys=keys, priority=Priorities.PREFETCH
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # prefetch priority
        return await self.pq_executor.submit_job(
            self._batched_get, keys=keys, priority=Priorities.PREFETCH
        )

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
        self.client.close()
        logger.info("Closed the RESP connection")


class RedisConnector(RemoteConnector):
    """
    The remote url should start with "redis://" and only have one host-port pair
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        # set a large max
        self.max_connections = 150
        # redis will crash if we have more than max_connections connections
        self.sem = asyncio.Semaphore(self.max_connections)
        self.pool = redis.ConnectionPool.from_url(
            url, max_connections=self.max_connections
        )
        self.connection = redis.Redis.from_pool(self.pool)
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.pq_executor = AsyncPQExecutor(loop)

    async def _exists(self, key: CacheEngineKey) -> bool:
        async with self.sem:
            return bool(await self.connection.exists(key.to_string() + "metadata"))

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        return bool(future.result())

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        async with self.sem:
            metadata_bytes = await self.connection.get(key_str + "metadata")

            if metadata_bytes is None:
                return None

            assert not inspect.isawaitable(metadata_bytes)

            metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))

            memory_obj = self.local_cpu_backend.allocate(
                metadata.shapes,
                metadata.dtypes,
                metadata.fmt,
            )
            if memory_obj is None:
                logger.warning("Failed to allocate memory during remote receive")
                return None

            # TODO(Jiayi): Find a way to do `get` inplace
            kv_bytes = await self.connection.get(key_str + "kv_bytes")
        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # TODO (Jiayi): We might need a way to better handle
            # consistency issues.
            # TODO (Jiayi): A better way is to aggregate metadata
            # and kv cache in one key.
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by redis."
            )
            async with self.sem:
                await self.connection.delete(key_str + "metadata")
            return None

        if isinstance(memory_obj.byte_array, memoryview):
            view = memory_obj.byte_array
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(memory_obj.byte_array)

        if isinstance(kv_bytes, (bytes, bytearray)):
            view[: metadata.length] = kv_bytes
        elif isinstance(kv_bytes, str):
            converted = kv_bytes.encode("utf-8")
            view[: metadata.length] = converted
        else:
            converted = bytes(kv_bytes)
            view[: metadata.length] = converted

        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    def support_batched_put(self) -> bool:
        return True

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        # calling self.put will create a circular dependency
        await asyncio.gather(
            *(
                self._put(key, memory_obj)
                for key, memory_obj in zip(keys, memory_objs, strict=False)
            )
        )

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        await self.pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shapes = memory_obj.get_shapes()
        kv_dtypes = memory_obj.get_dtypes()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), kv_shapes, kv_dtypes, memory_format
        ).serialize()

        key_str = key.to_string()
        # kv bytes needs to be set first to avoid race condition
        async with self.sem:
            await self.connection.set(key_str + "kv_bytes", kv_bytes)
            await self.connection.set(key_str + "metadata", metadata_bytes)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
        await self.connection.close()
        logger.info("Closed the redis connection")

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
            async with self.sem:
                if not await self.connection.exists(key.to_string() + "metadata"):
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

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # calling self.get will create a circular dependency
        results = await asyncio.gather(*(self._get(key) for key in keys))
        return [r for r in results if r is not None]

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )


class RedisSentinelConnector(RemoteConnector):
    """
    Uses redis.Sentinel to connect to a Redis cluster.
    The hosts are specified in the config file, started with "redis-sentinel://"
    and separated by commas.

    Example:
        remote_url: "redis-sentinel://localhost:26379,localhost:26380,localhost:26381"

    Extra environment variables:
    - REDIS_SERVICE_NAME (required) -- service name for redis.
    - REDIS_TIMEOUT (optional) -- Timeout in seconds, default is 1 if not set
    """

    ENV_REDIS_TIMEOUT = "REDIS_TIMEOUT"
    ENV_REDIS_SERVICE_NAME = "REDIS_SERVICE_NAME"

    def __init__(
        self,
        hosts_and_ports: List[Tuple[str, int]],
        username: str,
        password: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        # Get service name
        match os.environ.get(self.ENV_REDIS_SERVICE_NAME):
            case None:
                logger.warning(
                    f"Environment variable {self.ENV_REDIS_SERVICE_NAME} is "
                    f"not found, using default value 'redismaster'"
                )
                service_name = "redismaster"
            case value:
                service_name = value

        timeout: float = -1000.0

        # Get timeout
        match os.environ.get(self.ENV_REDIS_TIMEOUT):
            case None:
                timeout = 1
            case value:
                timeout = float(value)

        logger.info(f"Host and ports: {hosts_and_ports}")
        self.sentinel = redis.Sentinel(hosts_and_ports, socket_timeout=timeout)
        self.master = self.sentinel.master_for(
            service_name, socket_timeout=timeout, username=username, password=password
        )
        self.slave = self.sentinel.slave_for(
            service_name, socket_timeout=timeout, username=username, password=password
        )

        self.local_cpu_backend = local_cpu_backend

    async def exists(self, key: CacheEngineKey) -> bool:
        return bool(self.slave.exists(key.to_string() + "metadata"))

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return bool(self.slave.exists(key.to_string() + "metadata"))

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        metadata_bytes = self.slave.get(key_str + "metadata")

        if metadata_bytes is None:
            return None

        assert not inspect.isawaitable(metadata_bytes)

        metadata = RemoteMetadata.deserialize(metadata_bytes)

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shapes,
            metadata.dtypes,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        # TODO(Jiayi): Find a way to do `get` inplace
        kv_bytes = self.slave.get(key_str + "kv_bytes")

        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # TODO (Jiayi): We might need a way to better handle
            # consistency issues.
            # TODO (Jiayi): A background sweeper might be better
            # for the sake of performance.
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by redis."
            )
            self.master.delete(key_str + "metadata")
            return None

        if isinstance(memory_obj.byte_array, memoryview):
            view = memory_obj.byte_array
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(memory_obj.byte_array)

        if isinstance(kv_bytes, (bytes, bytearray)):
            view[0 : metadata.length] = kv_bytes
        elif isinstance(kv_bytes, str):
            converted = kv_bytes.encode("utf-8")
            view[0 : metadata.length] = converted
        else:
            converted = bytes(kv_bytes)
            view[0 : metadata.length] = converted

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shapes = memory_obj.get_shapes()
        kv_dtypes = memory_obj.get_dtypes()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), kv_shapes, kv_dtypes, memory_format
        ).serialize()

        key_str = key.to_string()
        # kv bytes needs to be set first to avoid race condition
        self.master.set(key_str + "kv_bytes", kv_bytes)
        self.master.set(key_str + "metadata", metadata_bytes)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.master.close()
        self.slave.close()


class RedisClusterConnector(RemoteConnector):
    """
    The remote url starts with "redis-cluster:// and can include one or
    multiple hosts:ports, separated by commas.

    Example:
        remote_url: "redis-cluster://host1:7000,host2:7000,host3:7000"

    Extra environment variables:
    - REDIS_TIMEOUT (optional) -- Timeout in seconds, default is 1 if not set
    """

    def __init__(
        self,
        hosts_and_ports: List[Tuple[str, int]],
        username: str,
        password: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        # Convert hosts_and_ports to startup_nodes format expected by RedisCluster
        startup_nodes = [ClusterNode(h, p) for (h, p) in hosts_and_ports]

        # set a large max
        self.max_connections = 150
        # redis will crash if we have more than max_connections connections
        self.sem = asyncio.Semaphore(self.max_connections)

        # Initialize cluster connection
        self.cluster = RedisCluster(
            startup_nodes=startup_nodes,
            username=username,
            password=password,
            max_connections=self.max_connections,
            decode_responses=False,
        )
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.pq_executor = AsyncPQExecutor(loop)

    async def _exists(self, key: CacheEngineKey) -> bool:
        async with self.sem:
            return bool(await self.cluster.exists(key.to_string() + "metadata"))

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        return bool(future.result())

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        async with self.sem:
            metadata_bytes = await self.cluster.get(key_str + "metadata")

            if metadata_bytes is None:
                return None

            assert not inspect.isawaitable(metadata_bytes)

            metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))

            memory_obj = self.local_cpu_backend.allocate(
                metadata.shapes,
                metadata.dtypes,
                metadata.fmt,
            )
            if memory_obj is None:
                logger.warning("Failed to allocate memory during remote receive")
                return None

            # TODO(Jiayi): Find a way to do `get` inplace
            kv_bytes = await self.cluster.get(key_str + "kv_bytes")

        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # TODO (Jiayi): We might need a way to better handle
            # consistency issues.
            # TODO (Jiayi): A better way is to aggregate metadata
            # and kv cache in one key.
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by redis."
            )
            async with self.sem:
                await self.cluster.delete(key_str + "metadata")
            return None

        if isinstance(memory_obj.byte_array, memoryview):
            view = memory_obj.byte_array
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(memory_obj.byte_array)

        if isinstance(kv_bytes, (bytes, bytearray)):
            view[: metadata.length] = kv_bytes
        elif isinstance(kv_bytes, str):
            converted = kv_bytes.encode("utf-8")
            view[: metadata.length] = converted
        else:
            converted = bytes(kv_bytes)
            view[: metadata.length] = converted

        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    def support_batched_put(self) -> bool:
        return True

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        # calling self.put will create a circular dependency
        await asyncio.gather(
            *(
                self._put(key, memory_obj)
                for key, memory_obj in zip(keys, memory_objs, strict=False)
            )
        )

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        await self.pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shapes = memory_obj.get_shapes()
        kv_dtypes = memory_obj.get_dtypes()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), kv_shapes, kv_dtypes, memory_format
        ).serialize()

        key_str = key.to_string()
        # kv bytes needs to be set first to avoid race condition
        async with self.sem:
            await self.cluster.set(key_str + "kv_bytes", kv_bytes)
            await self.cluster.set(key_str + "metadata", metadata_bytes)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
        await self.cluster.close()
        logger.info("Closed the redis cluster connection")

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
            async with self.sem:
                if not await self.cluster.exists(key.to_string() + "metadata"):
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

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # calling self.get will create a circular dependency
        results = await asyncio.gather(*(self._get(key) for key in keys))
        return [r for r in results if r is not None]

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )
