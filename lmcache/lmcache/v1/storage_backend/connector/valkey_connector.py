# SPDX-License-Identifier: Apache-2.0
"""
ValkeyConnector — high-throughput Valkey connector using the GLIDE sync
client (standalone or cluster) with a ThreadPoolExecutor and per-thread clients.

This replaces the legacy async ValkeyConnector / ValkeyClusterConnector
that used the async GLIDE client with 2-key storage.  The implementation
is shared with ``sync_valkey_connector.SyncValkeyConnector`` (which
registers on the ``valkey-sync://`` scheme as a backward-compat alias).

Design choices:
- The glide client lifecycle and worker thread pool live in the shared
  :class:`~lmcache.v1.storage_backend.valkey.worker_pool.ValkeyWorkerPool`,
  which is also used by the MP-mode ``ValkeyL2Adapter`` so both paths
  share one implementation of zero-copy GET/SET/EXISTS/DELETE.
- Direct ``memoryview`` access to pinned CPU memory — no shared-memory
  arena or cross-process copies needed since threads share the parent's
  address space.
- Single-key storage (like RESPConnector) to halve Valkey round-trips
  compared to the legacy 2-key metadata/kv_bytes split.
- Priority scheduling via ``AsyncPQExecutor`` (PEEK > PREFETCH > GET > PUT)
  ensures latency-sensitive lookups are not delayed behind bulk writes,
  matching the priority scheme used by ``RESPConnector``.

Migration notes from the old ValkeyConnector:
- Standalone mode (default) uses ``GlideClient`` and supports ``database_id``.
- Cluster mode (``valkey_mode: "cluster"``) uses ``GlideClusterClient`` which
  auto-discovers cluster topology from a single seed node.

Requires the ``valkey-glide-sync`` package (``>= 2.3.0``) for the
``glide_sync`` module; the plain ``valkey-glide`` package is async-only.
"""

# Standard
from enum import IntEnum, auto
from typing import List, Optional
import asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.valkey.worker_pool import (
    DEFAULT_CONNECTION_TIMEOUT_SECS,
    DEFAULT_REQUEST_TIMEOUT_SECS,
    GET_MISS,
    ValkeyWorkerPool,
)

logger = init_logger(__name__)

#: Default key TTL (seconds) used when the TTL feature flag is enabled but no
#: explicit ``valkey_ttl_sec`` is configured (24 hours). The request/connection
#: timeout defaults are re-exported from ``worker_pool`` (imported above) so
#: ``valkey_adapter`` can import all three from this module.
DEFAULT_TTL_SECS: int = 86400


class Priorities(IntEnum):
    """Operation priorities for the ``AsyncPQExecutor``.

    Lower numeric value = higher priority.  Matches the scheme used by
    ``RESPConnector`` so that exists/peek checks run before bulk writes.
    """

    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


class ValkeyConnector(RemoteConnector):
    """High-throughput Valkey connector using GLIDE sync cluster client with
    per-thread clients, ThreadPoolExecutor, and priority scheduling.

    Uses N worker threads, each with its own GLIDE sync client, and
    direct memoryview access for zero-copy data transfer.  Single-key
    storage halves Valkey round-trips compared to the legacy 2-key split.

    Operations are dispatched through an ``AsyncPQExecutor`` with priority
    levels (PEEK > PREFETCH > GET > PUT) so that latency-sensitive lookups
    are not delayed behind bulk writes.

    Args:
        host: Valkey server hostname.
        port: Valkey server port.
        loop: Asyncio event loop (used for PQ executor and wrap_future).
        local_cpu_backend: Backend for allocating CPU memory objects.
        num_workers: Number of worker threads (default 8).
        username: Valkey authentication username.
        password: Valkey authentication password.
        request_timeout: Timeout in seconds for requests and
            Future.result() calls (default 5).
        connection_timeout: Timeout in seconds for initial client
            connections (default 10).
        tls_enable: Whether to use TLS for Valkey connections.
        cluster_mode: If True, use GlideClusterClient; else GlideClient.
        database_id: Database ID for standalone mode (ignored in cluster).
        ttl_seconds: If set, keys are written with this expiry (seconds) so
            ``volatile-*`` eviction policies can reclaim them. ``None``
            (default) disables TTL.
    """

    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        num_workers: int = 8,
        username: str = "",
        password: str = "",
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECS,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT_SECS,
        tls_enable: bool = False,
        cluster_mode: bool = False,
        database_id: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ):
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        self.host = host
        self.port = port
        self.num_workers = num_workers
        self._request_timeout = request_timeout
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self._pool = ValkeyWorkerPool(
            addresses=[(host, port)],
            num_workers=num_workers,
            username=username,
            password=password,
            request_timeout=request_timeout,
            connection_timeout=connection_timeout,
            tls_enable=tls_enable,
            cluster_mode=cluster_mode,
            database_id=database_id,
            ttl_seconds=ttl_seconds,
        )
        self._pq_executor = AsyncPQExecutor(loop)

    # ── EXISTS ───────────────────────────────────────────────────────────

    async def _exists(self, key: CacheEngineKey) -> bool:
        """Internal: check if a key exists in Valkey."""
        return await asyncio.wrap_future(self._pool.submit_exists(key.to_string()))

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if a key exists in Valkey."""
        return await self._pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Synchronously check if a key exists in Valkey."""
        return self._pool.submit_exists(key.to_string()).result(
            timeout=self._request_timeout
        )

    # ── GET ──────────────────────────────────────────────────────────────

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Internal: retrieve a memory object from Valkey by key."""
        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes, self.meta_dtypes, self.meta_fmt
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        # The pool casts the buffer to a byte view internally, so pass
        # ``byte_array`` straight through.
        try:
            nbytes = await asyncio.wrap_future(
                self._pool.submit_get_into(key.to_string(), memory_obj.byte_array)
            )
        except Exception:
            memory_obj.ref_count_down()
            raise

        if nbytes <= GET_MISS:
            memory_obj.ref_count_down()
            return None
        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Retrieve a memory object from Valkey by key."""
        return await self._pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    # ── PUT ──────────────────────────────────────────────────────────────

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Internal: store a memory object in Valkey.

        Note: This method does NOT call ``ref_count_down()`` on the memory
        object.  Reference counting is managed by the caller
        (``RemoteBackend.submit_put_task``), which increments before
        submitting and decrements in the done callback.
        """
        await asyncio.wrap_future(
            self._pool.submit_set(key.to_string(), memory_obj.byte_array)
        )

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Store a memory object in Valkey."""
        await self._pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    # ── BATCHED PUT ──────────────────────────────────────────────────────

    def support_batched_put(self) -> bool:
        """Returns True — batched put is supported."""
        return True

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> None:
        """Internal: store multiple memory objects in Valkey in parallel."""
        n = len(keys)
        key_strs = [k.to_string() for k in keys]

        futures = [
            self._pool.submit_set(key_strs[i], memory_objs[i].byte_array)
            for i in range(n)
        ]
        wrapped = [asyncio.wrap_future(f) for f in futures]
        await asyncio.gather(*wrapped)

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> None:
        """Store multiple memory objects in Valkey in parallel."""
        await self._pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    # ── BATCHED GET ──────────────────────────────────────────────────────

    def support_batched_get(self) -> bool:
        """Returns True — batched get is supported."""
        return True

    async def _batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """Internal: retrieve multiple memory objects from Valkey in parallel.

        Note: Once an allocation failure occurs, all subsequent slots are
        set to ``None`` (intentional — memory pressure means further
        allocations would also fail).
        """
        n = len(keys)
        key_strs = [k.to_string() for k in keys]

        memory_objs: List[Optional[MemoryObj]] = []
        dst_bufs: List[Optional[memoryview]] = []
        alloc_failed = False

        for _ in keys:
            if alloc_failed:
                memory_objs.append(None)
                dst_bufs.append(None)
                continue

            mobj = self.local_cpu_backend.allocate(
                self.meta_shapes, self.meta_dtypes, self.meta_fmt
            )
            if mobj is None:
                logger.warning(
                    "Failed to allocate memory during batched remote receive"
                )
                alloc_failed = True
                memory_objs.append(None)
                dst_bufs.append(None)
                continue

            memory_objs.append(mobj)
            # The pool casts the buffer to a byte view internally.
            dst_bufs.append(mobj.byte_array)

        # Submit GET futures for allocated slots; track which indices have
        # real futures vs None (allocation failed).
        live_indices: List[int] = []
        live_futures: List[asyncio.Future] = []
        for i in range(n):
            if memory_objs[i] is not None and dst_bufs[i] is not None:
                live_indices.append(i)
                live_futures.append(
                    asyncio.wrap_future(
                        self._pool.submit_get_into(
                            key_strs[i],
                            dst_bufs[i],  # type: ignore[arg-type]
                        )
                    )
                )

        try:
            results = await asyncio.gather(*live_futures)
            for idx, nbytes in zip(live_indices, results, strict=True):
                if nbytes <= GET_MISS:
                    memory_objs[idx].ref_count_down()  # type: ignore[union-attr]
                    memory_objs[idx] = None
        except Exception:
            for mobj in memory_objs:
                if mobj is not None:
                    mobj.ref_count_down()
            raise

        return memory_objs

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """Retrieve multiple memory objects from Valkey in parallel."""
        return await self._pq_executor.submit_job(
            self._batched_get, keys=keys, priority=Priorities.GET
        )

    # ── BATCHED CONTAINS ─────────────────────────────────────────────────

    def support_batched_contains(self) -> bool:
        """Returns True — synchronous batched contains is supported."""
        return True

    def _count_consecutive_exists(self, keys: List[CacheEngineKey]) -> int:
        """Check how many consecutive keys exist (prefix match).

        Fans out individual EXISTS checks across the thread pool for
        parallel round-trips: wall-clock time is roughly
        ``ceil(N / num_workers) * RTT`` instead of ``N * RTT``.
        """
        key_strs = [k.to_string() for k in keys]
        futures = [self._pool.submit_exists(k) for k in key_strs]
        for i, fut in enumerate(futures):
            if not fut.result(timeout=self._request_timeout):
                return i
        return len(futures)

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        """Synchronously check how many consecutive keys exist."""
        return self._count_consecutive_exists(keys)

    def support_batched_async_contains(self) -> bool:
        """Returns True — async batched contains is supported."""
        return True

    async def _batched_async_contains(
        self,
        keys: List[CacheEngineKey],
    ) -> int:
        """Internal: asynchronously check how many consecutive keys exist.

        Fans out individual EXISTS checks across the thread pool.
        """
        key_strs = [k.to_string() for k in keys]
        wrapped = [asyncio.wrap_future(self._pool.submit_exists(k)) for k in key_strs]
        results = await asyncio.gather(*wrapped)
        for i, r in enumerate(results):
            if not r:
                return i
        return len(results)

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Asynchronously check how many consecutive keys exist."""
        return await self._pq_executor.submit_job(
            self._batched_async_contains,
            keys=keys,
            priority=Priorities.PREFETCH,
        )

    def support_batched_get_non_blocking(self) -> bool:
        """Returns True — non-blocking batched get is supported."""
        return True

    async def _batched_get_non_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        """Internal: non-blocking batched get returning the consecutive prefix.

        Only the consecutive prefix of non-None memory objects (from the
        beginning) is returned.  Once a ``None`` (missing key or allocation
        failure) is encountered, all subsequent objects — even if they were
        successfully retrieved — are released and excluded.  This matches
        the base-class contract.
        """
        all_results = await self._batched_get(keys)

        prefix: List[MemoryObj] = []
        found_failure = False
        for result in all_results:
            if found_failure:
                if result is not None:
                    result.ref_count_down()
            elif result is not None:
                prefix.append(result)
            else:
                found_failure = True

        return prefix

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        """Non-blocking batched get returning the consecutive prefix."""
        return await self._pq_executor.submit_job(
            self._batched_get_non_blocking,
            keys=keys,
            priority=Priorities.PREFETCH,
        )

    async def list(self) -> List[str]:
        """List all keys (not implemented)."""
        return []

    async def close(self) -> None:
        """Shut down the PQ executor and the thread pool."""
        await self._pq_executor.shutdown_async(wait=True)
        self._pool.close()
        logger.info("Closed ValkeyConnector")
