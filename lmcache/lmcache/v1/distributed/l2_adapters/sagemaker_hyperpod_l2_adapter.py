# SPDX-License-Identifier: Apache-2.0
"""SageMaker HyperPod ai-toolkit L2 adapter for LMCache MP mode."""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Optional, cast
import asyncio
import threading
import time

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.distributed.l2_adapters.sagemaker_hyperpod_client import (
    SageMakerHyperPodClient,
    SageMakerHyperPodLease,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


def _wire_key(key: ObjectKey) -> str:
    """Serialize an MP ``ObjectKey`` to an ai-toolkit object name.

    Unsalted::

        <model_name>@<kv_rank_hex8>@<object_group_id_hex>@<chunk_hash_hex>

    Salted (trailing ``cache_salt``)::

        <model_name>@<kv_rank_hex8>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>
    """
    base = (
        f"{key.model_name}@{key.kv_rank:08x}"
        f"@{key.object_group_id:x}@{key.chunk_hash.hex()}"
    )
    return f"{base}@{key.cache_salt}" if key.cache_salt else base


class SageMakerHyperPodL2AdapterConfig(L2AdapterConfigBase):
    """Configuration for the SageMaker HyperPod ai-toolkit L2 adapter."""

    def __init__(
        self,
        url: str,
        bucket: str = "lmcache",
        shared_memory_name: str = "shared_memory",
        max_concurrent_requests: int = 100,
        max_connections: int = 256,
        max_connections_per_host: int = 128,
        timeout_ms: int = 5000,
        lease_wait_timeout_ms: int = 1000,
        lease_ttl_ms: int = 30000,
        put_stream_chunk_bytes: int = 64 * 1024,
        max_lease_size_mb: float | None = None,
        use_https: bool = False,
    ) -> None:
        """Initialize and validate adapter configuration.

        Args:
            url: ai-toolkit daemon URL (``sagemaker-hyperpod://host:port``).
            bucket: ai-toolkit cache namespace.
            shared_memory_name: POSIX shared-memory segment name without the
                ``/dev/shm/`` prefix.
            max_concurrent_requests: Maximum concurrent HTTP requests.
            max_connections: Maximum total pooled HTTP connections.
            max_connections_per_host: Maximum pooled connections per host.
            timeout_ms: HTTP transport timeout in milliseconds (PUT and
                lease release).
            lease_wait_timeout_ms: Budget the daemon may spend holding a
                lease request before answering; bounds worst-case lookup
                latency.
            lease_ttl_ms: Server-side lease lifetime in milliseconds.
            put_stream_chunk_bytes: HTTP PUT streaming chunk size.
            max_lease_size_mb: Optional upper bound for accepted leases.
            use_https: Use HTTPS instead of HTTP for daemon requests.

        Raises:
            ValueError: If any field is empty, non-positive, or has the
                wrong type, or if ``url`` uses an unsupported scheme.
        """
        super().__init__()
        if not isinstance(url, str) or not url:
            raise ValueError("url must be a non-empty string")
        if not isinstance(use_https, bool):
            raise ValueError("use_https must be a boolean")
        SageMakerHyperPodClient.normalize_url(url, use_https=use_https)
        if not isinstance(bucket, str) or not bucket:
            raise ValueError("bucket must be a non-empty string")
        if not isinstance(shared_memory_name, str) or not shared_memory_name:
            raise ValueError("shared_memory_name must be a non-empty string")
        for name, value in (
            ("max_concurrent_requests", max_concurrent_requests),
            ("max_connections", max_connections),
            ("max_connections_per_host", max_connections_per_host),
            ("timeout_ms", timeout_ms),
            ("lease_wait_timeout_ms", lease_wait_timeout_ms),
            ("lease_ttl_ms", lease_ttl_ms),
            ("put_stream_chunk_bytes", put_stream_chunk_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if max_lease_size_mb is not None and (
            isinstance(max_lease_size_mb, bool)
            or not isinstance(max_lease_size_mb, (int, float))
            or max_lease_size_mb <= 0
        ):
            raise ValueError("max_lease_size_mb must be positive when set")

        self.url = url
        self.bucket = bucket
        self.shared_memory_name = shared_memory_name
        self.max_concurrent_requests = max_concurrent_requests
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.timeout_ms = timeout_ms
        self.lease_wait_timeout_ms = lease_wait_timeout_ms
        self.lease_ttl_ms = lease_ttl_ms
        self.put_stream_chunk_bytes = put_stream_chunk_bytes
        self.max_lease_size_mb = max_lease_size_mb
        self.use_https = use_https

    @classmethod
    def from_dict(cls, d: dict) -> "SageMakerHyperPodL2AdapterConfig":
        """Parse an adapter configuration from a JSON-derived dictionary.

        Args:
            d: Parsed ``--l2-adapter`` JSON object.

        Returns:
            A validated configuration instance.

        Raises:
            ValueError: If ``d`` contains an ``eviction`` block (ai-toolkit
                owns L2 eviction) or any field fails validation.
        """
        if d.get("eviction") is not None:
            raise ValueError(
                "sagemaker-hyperpod does not support LMCache-owned eviction"
            )
        return cls(
            url=d.get("url", ""),
            bucket=d.get("bucket", "lmcache"),
            shared_memory_name=d.get("shared_memory_name", "shared_memory"),
            max_concurrent_requests=d.get("max_concurrent_requests", 100),
            max_connections=d.get("max_connections", 256),
            max_connections_per_host=d.get("max_connections_per_host", 128),
            timeout_ms=d.get("timeout_ms", 5000),
            lease_wait_timeout_ms=d.get("lease_wait_timeout_ms", 1000),
            lease_ttl_ms=d.get("lease_ttl_ms", 30000),
            put_stream_chunk_bytes=d.get(
                "put_stream_chunk_bytes",
                64 * 1024,
            ),
            max_lease_size_mb=d.get("max_lease_size_mb"),
            use_https=d.get("use_https", False),
        )

    @classmethod
    def help(cls) -> str:
        """Return CLI help for the adapter's JSON fields."""
        return (
            "SageMaker HyperPod L2 adapter fields:\n"
            "- url (str, required): sagemaker-hyperpod://host:port\n"
            "- bucket (str, default 'lmcache'): cache namespace\n"
            "- shared_memory_name (str, default 'shared_memory')\n"
            "- max_concurrent_requests (int, default 100)\n"
            "- max_connections (int, default 256)\n"
            "- max_connections_per_host (int, default 128)\n"
            "- timeout_ms (int, default 5000): HTTP transport timeout\n"
            "- lease_wait_timeout_ms (int, default 1000): how long the daemon\n"
            "  may hold a lease request before answering\n"
            "- lease_ttl_ms (int, default 30000)\n"
            "- put_stream_chunk_bytes (int, default 65536)\n"
            "- max_lease_size_mb (float, optional)\n"
            "- use_https (bool, default false): HTTPS instead of HTTP\n"
            "LMCache-owned eviction, delete, and list are not supported."
        )


class SageMakerHyperPodL2Adapter(L2AdapterInterface):
    """Asynchronous MP L2 adapter backed by the node-local ai-toolkit daemon."""

    def __init__(self, config: SageMakerHyperPodL2AdapterConfig) -> None:
        """Create the adapter and start its asynchronous worker loop.

        Args:
            config: Validated adapter configuration.

        Raises:
            RuntimeError: If the worker event loop does not start in time or
                the ai-toolkit shared-memory segment cannot be opened.
        """
        super().__init__()
        self._config = config
        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()
        self._completed_stores: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookups: dict[L2TaskId, Bitmap] = {}
        self._completed_loads: dict[L2TaskId, Bitmap] = {}
        self._leases: dict[ObjectKey, list[SageMakerHyperPodLease]] = {}
        # Stored sizes for byte-accounting dedup. NOTE: grows unbounded
        # (delete is never called); shared limitation across remote adapters.
        self._key_sizes: dict[ObjectKey, int] = {}
        self._next_task_id: L2TaskId = 0
        self._lock = threading.Lock()
        self._closed = False
        self._inflight: set[Future[Any]] = set()
        self._releasing_leases: set[SageMakerHyperPodLease] = set()

        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="sagemaker-hyperpod-l2",
        )
        self._loop_thread.start()
        if not self._loop_ready.wait(timeout=5):
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            if not self._loop_thread.is_alive():
                self._loop.close()
            self._store_efd.close()
            self._lookup_efd.close()
            self._load_efd.close()
            raise RuntimeError("timed out starting SageMaker HyperPod L2 event loop")

        try:
            self._client = asyncio.run_coroutine_threadsafe(
                self._create_client(config),
                self._loop,
            ).result(timeout=5)
        except Exception:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            if not self._loop_thread.is_alive():
                self._loop.close()
            self._store_efd.close()
            self._lookup_efd.close()
            self._load_efd.close()
            raise

        logger.info(
            "SageMakerHyperPodL2Adapter ready: url=%s bucket=%s shared_memory=%s",
            config.url,
            config.bucket,
            config.shared_memory_name,
        )

    def get_store_event_fd(self) -> int:
        """Return the store-completion event descriptor."""
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the lookup-completion event descriptor."""
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        """Return the load-completion event descriptor."""
        return self._load_efd.fileno()

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a non-blocking batch store.

        Args:
            keys: Object keys to store.
            objects: L1-owned memory objects matching ``keys`` positionally.
                The caller must keep them alive until the task completes.

        Returns:
            A task ID whose result is popped through
            :meth:`pop_completed_store_tasks` after the store event fd fires.

        Raises:
            ValueError: If ``keys`` and ``objects`` lengths differ.
        """
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have equal lengths")
        task_id = self._allocate_task_id()
        if self._closed:
            self._complete_closed_store(task_id)
            return task_id

        keys_snapshot = list(keys)
        objects_snapshot = list(objects)
        sizes = [obj.get_size() for obj in objects_snapshot]
        future = asyncio.run_coroutine_threadsafe(
            self._store(keys_snapshot, objects_snapshot),
            self._loop,
        )
        self._track_future(
            future,
            lambda done: self._finish_store(
                task_id,
                keys_snapshot,
                sizes,
                done,
            ),
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Pop all completed store results.

        Returns:
            A mapping from task ID to :class:`L2StoreResult`. Each result is
            returned exactly once.
        """
        with self._lock:
            completed = self._completed_stores
            self._completed_stores = {}
        return completed

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> L2TaskId:
        """Submit a lookup that retains daemon leases for every hit.

        Args:
            keys: Object keys to look up.
            layout_desc: Unused; ai-toolkit leases are layout-independent.

        Returns:
            A task ID whose hit bitmap is queried through
            :meth:`query_lookup_and_lock_result` after the lookup event fd
            fires. Retained leases are held until :meth:`submit_unlock`.
        """
        del layout_desc
        task_id = self._allocate_task_id()
        if self._closed:
            with self._lock:
                self._completed_lookups[task_id] = Bitmap(len(keys))
            self._lookup_efd.notify()
            return task_id
        keys_snapshot = list(keys)
        future = asyncio.run_coroutine_threadsafe(
            self._lookup(keys_snapshot),
            self._loop,
        )
        self._track_future(
            future,
            lambda done: self._finish_lookup(task_id, keys_snapshot, done),
        )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Pop one completed lookup result, or return ``None``.

        Args:
            task_id: Task ID returned by :meth:`submit_lookup_and_lock_task`.

        Returns:
            The hit bitmap, or ``None`` when the task has not completed or
            the result was already popped.
        """
        with self._lock:
            return self._completed_lookups.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Asynchronously release one retained lease for each key occurrence.

        Args:
            keys: Keys previously locked by a lookup. A key occurring N times
                releases up to N retained leases. Failed releases are retried
                until the lease TTL, after which the daemon expires them.
        """
        leases: list[SageMakerHyperPodLease] = []
        with self._lock:
            for key in keys:
                key_leases = self._leases.get(key)
                if not key_leases:
                    continue
                leases.append(key_leases.pop())
                if not key_leases:
                    self._leases.pop(key, None)
        self._schedule_release(leases)

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a non-blocking load into caller-owned buffers.

        Args:
            keys: Object keys to load, normally locked by a prior lookup.
            objects: Destination memory objects matching ``keys``
                positionally. The caller must keep them alive until the task
                completes.

        Returns:
            A task ID whose per-key success bitmap is queried through
            :meth:`query_load_result` after the load event fd fires.

        Raises:
            ValueError: If ``keys`` and ``objects`` lengths differ.
        """
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have equal lengths")
        task_id = self._allocate_task_id()
        if self._closed:
            with self._lock:
                self._completed_loads[task_id] = Bitmap(len(keys))
            self._load_efd.notify()
            return task_id
        keys_snapshot = list(keys)
        objects_snapshot = list(objects)
        future = asyncio.run_coroutine_threadsafe(
            self._load(keys_snapshot, objects_snapshot),
            self._loop,
        )
        self._track_future(
            future,
            lambda done: self._finish_load(task_id, keys_snapshot, done),
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Pop one completed load result, or return ``None``.

        Args:
            task_id: Task ID returned by :meth:`submit_load_task`.

        Returns:
            The per-key success bitmap, or ``None`` when the task has not
            completed or the result was already popped.
        """
        with self._lock:
            return self._completed_loads.pop(task_id, None)

    def report_status(self) -> dict[str, object]:
        """Return adapter, shared-memory, and HTTP transport health.

        Returns:
            A dictionary with ``is_healthy`` (adapter open, worker loop
            alive, and backend healthy), adapter identity fields (``type``,
            ``url``, ``bucket``, ``shared_memory_name``), and the client's
            ``backend`` health report.
        """
        client_report = getattr(self._client, "report_status", None)
        backend = client_report() if callable(client_report) else {"is_healthy": True}
        return {
            "is_healthy": (
                not self._closed
                and self._loop_thread.is_alive()
                and bool(backend.get("is_healthy", False))
            ),
            "type": "sagemaker-hyperpod",
            "url": self._config.url,
            "bucket": self._config.bucket,
            "shared_memory_name": self._config.shared_memory_name,
            "backend": backend,
        }

    def close(self) -> None:
        """Cancel in-flight work, release leases, and close all resources.

        Shutdown is bounded: each remaining lease gets one best-effort
        release attempt and unreleased leases expire server-side after
        their TTL.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            inflight = list(self._inflight)

        for task in inflight:
            task.cancel()
        deadline = time.monotonic() + 5.0
        for task in inflight:
            try:
                task.result(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                pass

        with self._lock:
            leases = [lease for values in self._leases.values() for lease in values]
            leases.extend(self._releasing_leases)
            self._leases.clear()
            self._releasing_leases.clear()

        future = asyncio.run_coroutine_threadsafe(
            self._close_async(list(set(leases))),
            self._loop,
        )
        close_timeout = self._config.timeout_ms / 1000.0 + 1.0
        try:
            future.result(timeout=max(5.0, close_timeout))
        except Exception as exc:
            logger.warning("Error closing SageMaker HyperPod L2 client: %s", exc)

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)
        if self._loop_thread.is_alive():
            logger.error("SageMaker HyperPod L2 event loop did not stop")
        else:
            self._loop.close()
        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    async def _create_client(
        self,
        config: SageMakerHyperPodL2AdapterConfig,
    ) -> SageMakerHyperPodClient:
        return SageMakerHyperPodClient(
            url=config.url,
            bucket=config.bucket,
            shared_memory_name=config.shared_memory_name,
            max_concurrent_requests=config.max_concurrent_requests,
            max_connections=config.max_connections,
            max_connections_per_host=config.max_connections_per_host,
            timeout_ms=config.timeout_ms,
            lease_wait_timeout_ms=config.lease_wait_timeout_ms,
            lease_ttl_ms=config.lease_ttl_ms,
            put_stream_chunk_bytes=config.put_stream_chunk_bytes,
            max_lease_size_mb=config.max_lease_size_mb,
            use_https=config.use_https,
        )

    async def _store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> list[bool]:
        raw_results = await asyncio.gather(
            *(
                self._client.put(
                    _wire_key(key),
                    cast(memoryview, obj.byte_array),
                )
                for key, obj in zip(keys, objects, strict=True)
            ),
            return_exceptions=True,
        )
        return [
            bool(result) if not isinstance(result, BaseException) else False
            for result in raw_results
        ]

    async def _lookup(
        self,
        keys: list[ObjectKey],
    ) -> list[SageMakerHyperPodLease | None]:
        raw_results = await asyncio.gather(
            *(self._client.acquire_lease(_wire_key(key)) for key in keys),
            return_exceptions=True,
        )
        return [
            result
            if isinstance(result, SageMakerHyperPodLease) or result is None
            else None
            for result in raw_results
        ]

    async def _load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> tuple[list[bool], list[SageMakerHyperPodLease]]:
        lease_indices: dict[ObjectKey, int] = {}
        selected: list[SageMakerHyperPodLease | None] = []
        with self._lock:
            for key in keys:
                index = lease_indices.get(key, 0)
                available = self._leases.get(key, [])
                candidate = available[index] if index < len(available) else None
                # Expired retained lease: re-acquire a fresh one below.
                if candidate is not None and candidate.is_expired():
                    candidate = None
                selected.append(candidate)
                lease_indices[key] = index + 1

        # Acquire transient leases in parallel for keys without one.
        transient_leases: list[SageMakerHyperPodLease] = []
        missing = [index for index, lease in enumerate(selected) if lease is None]
        if missing:
            acquired = await asyncio.gather(
                *(self._client.acquire_lease(_wire_key(keys[i])) for i in missing),
                return_exceptions=True,
            )
            for index, result in zip(missing, acquired, strict=True):
                if isinstance(result, BaseException):
                    logger.warning("SageMaker HyperPod load lookup failed: %s", result)
                elif result is not None:
                    selected[index] = result
                    transient_leases.append(result)

        results: list[bool] = []
        for index, obj in enumerate(objects):
            lease = selected[index]
            try:
                loaded = lease is not None and self._client.copy_from_lease(
                    lease,
                    cast(memoryview, obj.byte_array),
                )
            except Exception as exc:
                logger.warning("SageMaker HyperPod shared-memory load failed: %s", exc)
                loaded = False
            results.append(loaded)
        return results, transient_leases

    def _schedule_release(self, leases: list[SageMakerHyperPodLease]) -> None:
        if not leases:
            return
        with self._lock:
            self._releasing_leases.update(leases)
            closed = self._closed
        if closed:
            return

        future = asyncio.run_coroutine_threadsafe(self._release_all(leases), self._loop)

        def _released(done: Future[Any]) -> None:
            try:
                done.result()
            except Exception as exc:
                logger.warning("SageMaker HyperPod lease release failed: %s", exc)
            finally:
                with self._lock:
                    self._releasing_leases.difference_update(leases)

        self._track_future(future, _released)

    async def _release_all(self, leases: list[SageMakerHyperPodLease]) -> list[bool]:
        return list(
            await asyncio.gather(
                *(self._release_with_retry(lease) for lease in leases),
                return_exceptions=False,
            )
        )

    async def _release_with_retry(self, lease: SageMakerHyperPodLease) -> bool:
        deadline = time.monotonic() + self._config.lease_ttl_ms / 1000.0
        while time.monotonic() < deadline:
            if lease.is_expired():
                # Already reclaimed server-side; nothing to release.
                return True
            if await self._client.release_lease(lease):
                return True
            await asyncio.sleep(min(0.1, self._config.lease_ttl_ms / 10000.0))
        logger.warning("Lease %s was left for server-side expiry", lease.lease_id)
        return False

    async def _close_async(self, leases: list[SageMakerHyperPodLease]) -> None:
        if leases:
            # Best-effort only: unreleased leases expire server-side.
            released = await asyncio.gather(
                *(self._client.release_lease(lease) for lease in leases),
                return_exceptions=True,
            )
            unreleased = sum(1 for result in released if result is not True)
            if unreleased:
                logger.warning(
                    "%d lease(s) left for server-side expiry at close", unreleased
                )
        await self._client.close()

    def _finish_store(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        sizes: list[int],
        future: Future[list[bool]],
    ) -> None:
        try:
            per_key_ok = future.result()
        except Exception as exc:
            logger.warning("SageMaker HyperPod store failed: %s", exc)
            per_key_ok = [False] * len(keys)
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        transferred = 0
        with self._lock:
            for key, size, ok in zip(keys, sizes, per_key_ok, strict=True):
                if not ok:
                    continue
                stored_keys.append(key)
                if key in self._key_sizes:
                    stored_sizes.append(0)
                else:
                    self._key_sizes[key] = size
                    stored_sizes.append(size)
                    transferred += size
            self._completed_stores[task_id] = L2StoreResult(
                all(per_key_ok),
                transferred,
            )
        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)
        self._store_efd.notify()

    def _finish_lookup(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        future: Future[list[SageMakerHyperPodLease | None]],
    ) -> None:
        try:
            leases = future.result()
        except Exception as exc:
            logger.warning("SageMaker HyperPod lookup failed: %s", exc)
            leases = [None] * len(keys)
        bitmap = Bitmap(len(keys))
        accessed: list[ObjectKey] = []
        with self._lock:
            for index, (key, lease) in enumerate(zip(keys, leases, strict=True)):
                if lease is None:
                    continue
                bitmap.set(index)
                self._leases.setdefault(key, []).append(lease)
                accessed.append(key)
            self._completed_lookups[task_id] = bitmap
        if accessed:
            self._notify_keys_accessed(accessed)
        self._lookup_efd.notify()

    def _finish_load(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        future: Future[tuple[list[bool], list[SageMakerHyperPodLease]]],
    ) -> None:
        transient_leases: list[SageMakerHyperPodLease] = []
        try:
            per_key_ok, transient_leases = future.result()
        except Exception as exc:
            logger.warning("SageMaker HyperPod load failed: %s", exc)
            per_key_ok = [False] * len(keys)
        bitmap = Bitmap(len(keys))
        accessed: list[ObjectKey] = []
        for index, (key, loaded) in enumerate(zip(keys, per_key_ok, strict=True)):
            if loaded:
                bitmap.set(index)
                accessed.append(key)
        with self._lock:
            self._completed_loads[task_id] = bitmap
        if accessed:
            self._notify_keys_accessed(accessed)
        self._load_efd.notify()
        self._schedule_release(transient_leases)

    def _track_future(
        self,
        future: Future[Any],
        callback: Callable[[Future[Any]], None] | None = None,
    ) -> None:
        with self._lock:
            self._inflight.add(future)

        def _done(done: Future[Any]) -> None:
            try:
                if callback is not None:
                    callback(done)
            finally:
                with self._lock:
                    self._inflight.discard(done)

        future.add_done_callback(_done)

    def _complete_closed_store(self, task_id: L2TaskId) -> None:
        with self._lock:
            self._completed_stores[task_id] = L2StoreResult(False, 0)
        self._store_efd.notify()

    def _allocate_task_id(self) -> L2TaskId:
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
        return task_id

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()


def _create_sagemaker_hyperpod_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional["L1MemoryDesc"] = None,
) -> L2AdapterInterface:
    """Create a SageMaker HyperPod L2 adapter from registered configuration."""
    del l1_memory_desc
    if not isinstance(config, SageMakerHyperPodL2AdapterConfig):
        raise TypeError(
            f"expected SageMakerHyperPodL2AdapterConfig, got {type(config).__name__}"
        )
    return SageMakerHyperPodL2Adapter(config)


register_l2_adapter_type(
    "sagemaker-hyperpod",
    SageMakerHyperPodL2AdapterConfig,
)
register_l2_adapter_factory(
    "sagemaker-hyperpod",
    _create_sagemaker_hyperpod_l2_adapter,
)
