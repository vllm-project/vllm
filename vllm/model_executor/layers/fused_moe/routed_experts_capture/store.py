# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-experts secondary-tier offload sidecar.

The scheduler-side offloaded-block buffer
(``RoutedExpertsManager.routed_experts_by_cpu_block``) follows the KV cache
through the offload tiers. The classes here let it cascade to / promote from a
secondary tier (disk/object/Mooncake/...) in lockstep with the KV blocks.

Routing is an MoE product, so it lives under ``fused_moe``; it only REUSES the
KV offload lifecycle (``kv_offload/``) as transport. The generic tier hook
(``BlockLifecycleObserver``) is defined in ``kv_offload/base.py``; everything
routing-specific — the backend interface, the pluggable factory, the built-in
filesystem backend, and the observer that bridges KV-tier events to a backend
— is here.
"""

from __future__ import annotations

import functools
import importlib
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from vllm.v1.kv_offload.base import BlockLifecycleObserver
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = logging.getLogger(__name__)


class RoutedExpertsSecondaryStore(ABC):
    """Persist / restore offloaded-block rows for a secondary tier.

    The routed-experts offload buffer lives in the CPU primary tier alongside
    KV. When KV blocks cascade to / promote from a secondary tier, the matching
    offloaded-block rows must follow so routing survives CPU eviction exactly
    as the KV bytes do. Implementations key each row by its ``OffloadKey``
    (block hash + group idx), mirroring the KV ``FileMapper`` layout.

    Rows are ``(factor, block_size, num_layers, top_k)`` arrays of the
    manager's expert-id dtype. ``persist`` may be asynchronous (the built-in
    ``fs`` backend enqueues writes onto a thread pool and returns, keeping the
    scheduler thread off the disk-IO critical path); ``restore`` is
    synchronous but must observe writes issued by a prior ``persist`` for the
    same keys (read-after-write), regardless of whether the write has reached
    its backing store yet.
    """

    @abstractmethod
    def persist(self, keys: Sequence[bytes], rows: np.ndarray) -> None:
        """Write one row per key. ``rows[i]`` corresponds to ``keys[i]``."""

    @abstractmethod
    def restore(self, keys: Sequence[bytes]) -> np.ndarray | None:
        """Read rows for keys, stacked in order.

        Returns ``None`` if any key is missing (the caller then leaves the
        offloaded-block rows untouched — the connector would not have issued a
        load for an absent block).
        """

    def prefetch(self, keys: Sequence[bytes]) -> None:
        """Hint that ``restore(keys)`` is coming soon; warm a read-ahead cache.

        Called when the matching KV blocks begin promoting (secondary ->
        primary), so a backend can overlap its read with the KV-byte transfer
        and serve the subsequent ``restore`` from memory instead of blocking
        the scheduler on disk. Default no-op: ``restore`` stays correct without
        it (it just reads synchronously). Idempotent and best-effort.
        """


class RoutedExpertsStoreContext(NamedTuple):
    """Inputs a ``RoutedExpertsSecondaryStore`` builder may need.

    Passed to every builder registered with ``RoutedExpertsStoreFactory`` so
    a backend (filesystem, object store, Mooncake, ...) can construct its
    store without the scheduler hard-coding any one implementation. The
    context is backend-agnostic: each backend reads whatever it needs (a
    ``root_dir``, an endpoint, credentials, ...) from ``tier_config``.

    Args:
        tier_config: The secondary-tier config dict from
            ``kv_connector_extra_config['secondary_tiers'][i]`` (includes
            ``type`` plus any backend-specific keys, e.g. ``root_dir`` for
            ``fs`` or endpoint / namespace for a remote store).
        offloading_spec: The resolved ``CPUOffloadingSpec`` (subclassed by
            ``TieringOffloadingSpec``); exposes ``block_size_factor``,
            ``vllm_config``, ``extra_config``, etc.
        row_shape: Offloaded-block row shape ``(factor, block_size, layers,
            top_k)``.
        dtype: Expert-id dtype of the offloaded-block rows.
    """

    tier_config: dict
    offloading_spec: "OffloadingSpec"
    row_shape: tuple[int, ...]
    dtype: np.dtype


class RoutedExpertsStoreFactory:
    """Registry mapping a secondary-tier ``type`` to a routing-store builder.

    Mirrors ``vllm.v1.kv_offload.tiering.factory.SecondaryTierFactory``: a
    backend registers a builder under the same ``type`` string its KV tier
    uses (e.g. ``"fs"``, ``"mooncake"``), and the scheduler looks it up
    instead of hard-coding any implementation. Registration is lazy (module
    path + factory name), so a backend's heavy imports load only when its tier
    is configured, and out-of-tree backends register without importing this
    module.

    To add a backend, implement the two-method ``RoutedExpertsSecondaryStore``
    contract (``persist`` / ``restore``, keyed by ``OffloadKey``) and register
    a builder::

        # my_pkg/mooncake_store.py
        class MooncakeRoutedExpertsStore(RoutedExpertsSecondaryStore):
            def __init__(self, ctx):
                self._store = open_mooncake(ctx.tier_config)  # by-key put/get
                self._shape, self._dtype = ctx.row_shape, ctx.dtype

            def persist(self, keys, rows):
                for k, row in zip(keys, rows):
                    self._store.put(k.hex(), row.tobytes())

            def restore(self, keys):
                bufs = [self._store.get(k.hex()) for k in keys]
                if any(not b for b in bufs):
                    return None
                return np.stack(
                    [np.frombuffer(b, self._dtype).reshape(self._shape) for b in bufs]
                )


        def build_store(ctx):
            return MooncakeRoutedExpertsStore(ctx)


        RoutedExpertsStoreFactory.register_store(
            "mooncake", "my_pkg.mooncake_store", "build_store"
        )

    A tier ``type`` with no registered builder just gets no routing sidecar (a
    warning is logged); KV still tiers normally.
    """

    _registry: dict[str, tuple[str, str]] = {}

    @classmethod
    def register_store(
        cls, tier_type: str, module_path: str, factory_name: str
    ) -> None:
        """Register a store-builder factory for a secondary-tier ``type``.

        Args:
            tier_type: Tier type string (must match the KV secondary tier's
                ``type``, e.g. ``"fs"``, ``"mooncake"``).
            module_path: Import path of the module holding the builder.
            factory_name: Name of the builder callable within that module;
                it takes a ``RoutedExpertsStoreContext`` and returns a
                ``RoutedExpertsSecondaryStore``.

        Raises:
            ValueError: If ``tier_type`` is already registered.
        """
        if tier_type in cls._registry:
            raise ValueError(
                f"Routed-experts store for tier '{tier_type}' is already registered."
            )
        cls._registry[tier_type] = (module_path, factory_name)

    @classmethod
    def is_registered(cls, tier_type: str) -> bool:
        return tier_type in cls._registry

    @classmethod
    def create(
        cls, tier_type: str, ctx: RoutedExpertsStoreContext
    ) -> RoutedExpertsSecondaryStore | None:
        """Build the store for ``tier_type``, or None if no builder is known."""
        entry = cls._registry.get(tier_type)
        if entry is None:
            return None
        module_path, factory_name = entry
        module = importlib.import_module(module_path)
        builder = getattr(module, factory_name)
        return builder(ctx)


class FileRoutedExpertsStore(RoutedExpertsSecondaryStore):
    """Disk-backed ``RoutedExpertsSecondaryStore`` (the built-in ``fs`` tier).

    One ``.re`` file per offloaded block, named from the block's
    ``OffloadKey`` under a per-rank directory, reusing the KV ``FileMapper``
    hashing/fan-out so routing files sit beside KV files and inherit the
    same parallel-layout isolation.

    Writes are asynchronous, mirroring the KV ``FileSystemTierManager``: the
    scheduler-thread ``persist`` only stashes a copy of each row in an
    in-memory ``_pending`` map and enqueues the actual disk write onto a
    ``DualQueueThreadPool``, then returns immediately. This keeps cascade
    events (which fire on the single scheduler thread inside the engine's
    busy loop) off the synchronous-IO critical path — a synchronous write
    there blocks every in-flight request and was measured to dominate the
    decode-step tail under offload.

    Read-after-write stays correct without waiting on disk: ``restore`` checks
    ``_pending`` first (a cascade immediately followed by a promotion of the
    same block reads the still-in-memory copy), falling back to disk only for
    keys whose write already drained. A completed write pops its key from
    ``_pending``.

    Reads are prefetched: when KV blocks start promoting, ``prefetch`` enqueues
    background disk reads into ``_prefetched``, overlapping the (256x larger)
    KV-byte transfer. The subsequent ``restore`` then serves from memory
    instead of blocking the scheduler on disk; a prefetch miss falls back to a
    synchronous read, so correctness never depends on the prefetch landing.

    IO is zero-copy, mirroring the KV ``store_block`` / ``load_block`` path:
    writes ``os.write`` the snapshot ndarray's byte-view directly (no
    ``.tobytes()``); reads ``os.readv`` straight into the destination row
    ndarray (no intermediate ``bytes`` / ``np.frombuffer``). Only the write
    snapshot copy is kept — required because the source CPU block is not
    ref-protected for the lifetime of the async routing write.
    """

    def __init__(
        self,
        root_dir: str,
        file_mapper,
        row_shape: tuple[int, ...],
        dtype: np.dtype,
        n_write_threads: int = 4,
        n_read_threads: int = 4,
    ) -> None:
        self._file_mapper = file_mapper
        self._row_shape = row_shape
        self._dtype = np.dtype(dtype)
        self._row_bytes = int(np.prod(row_shape)) * self._dtype.itemsize

        # In-memory copies of rows whose async write has not yet completed,
        # keyed by offload key. Protected by ``_cache_lock``. Guarantees
        # read-after-write without blocking the scheduler on disk.
        self._pending: dict[bytes, np.ndarray] = {}
        # Read-ahead cache filled by ``prefetch``; consumed (popped) by
        # ``restore``. Shares the lock with ``_pending``.
        self._prefetched: dict[bytes, np.ndarray] = {}
        self._cache_lock = threading.Lock()

        # Async IO pool, same primitive the KV fs tier uses: store-priority
        # threads drain writes (``persist``), load-priority threads drain
        # prefetch reads, neither starves the other.
        self._job_counter = 0
        self._closed = False
        self._pool = DualQueueThreadPool(
            n_read_threads=max(1, n_read_threads),
            n_write_threads=max(1, n_write_threads),
            thread_name_prefix="vllm_re_fs",
        )

    def _path(self, key: bytes) -> str:
        # FileMapper.get_file_name returns "<...>/<hash>.bin"; swap the
        # suffix so routing sidecars never collide with KV block files.
        return self._file_mapper.get_file_name(key)[: -len(".bin")] + ".re"

    def _read_row(self, path: str) -> np.ndarray | None:
        """Read one sidecar off disk, or ``None`` if absent.

        Zero-copy: ``os.readv`` fills a freshly-allocated row ndarray directly
        (via its byte-view), avoiding the intermediate ``bytes`` object and the
        ``np.frombuffer`` re-wrap. Loops to tolerate short reads. Raises on a
        size mismatch (corrupt / truncated file), matching ``restore``'s check.
        """
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            row = np.empty(self._row_shape, dtype=self._dtype)
            view = memoryview(row).cast("B")
            got = 0
            while got < self._row_bytes:
                chunk = os.readv(fd, [view[got:]])
                if chunk == 0:
                    break  # EOF before full row -> truncated
                got += chunk
        finally:
            os.close(fd)
        if got != self._row_bytes:
            raise RuntimeError(
                f"routed-experts sidecar {path} has {got} bytes, "
                f"expected {self._row_bytes}"
            )
        return row

    def _write_one(self, key: bytes, path: str, row: np.ndarray) -> None:
        """Write a single sidecar atomically, then drop it from pending.

        Runs on a pool worker thread. Zero-copy: ``os.write`` consumes the
        snapshot ndarray's byte-view directly (no ``.tobytes()`` copy), looping
        to tolerate short writes. The snapshot itself is required — the source
        CPU block may be reused once this returns, and the write runs later.
        Idempotent: an already-present file is left untouched. The pending
        entry is removed in a ``finally`` so a failed write does not leak
        memory (a later ``restore`` for that key then falls through to disk and
        reports a miss).
        """
        try:
            if os.path.exists(path):
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            payload = memoryview(row).cast("B")
            try:
                fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                try:
                    written = 0
                    while written < self._row_bytes:
                        written += os.write(fd, payload[written:])
                finally:
                    os.close(fd)
                os.replace(tmp, path)
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
        finally:
            with self._cache_lock:
                self._pending.pop(key, None)

    def _read_one(self, key: bytes, path: str) -> None:
        """Prefetch one sidecar into ``_prefetched``. Runs on a pool worker.

        A miss / error simply leaves nothing cached; the later ``restore``
        falls back to a synchronous read, so this is purely best-effort.
        """
        try:
            row = self._read_row(path)
        except Exception as exc:
            logger.warning("routed-experts prefetch read failed (%s): %s", path, exc)
            return
        if row is not None:
            with self._cache_lock:
                self._prefetched[key] = row

    def _reap_finished(self) -> None:
        """Drain the pool's finished-jobs queue so it can't grow unbounded.

        Per-task bookkeeping (``_pending`` pop on write, ``_prefetched`` fill on
        read) happens in the worker itself; this only discards the job-id
        records the pool accumulates as jobs complete.
        """
        self._pool.get_finished()

    def _next_job_id(self) -> int:
        self._job_counter += 1
        return self._job_counter

    def persist(self, keys: Sequence[bytes], rows: np.ndarray) -> None:
        self._reap_finished()
        tasks = []
        for key, row in zip(keys, rows):
            path = self._path(key)
            # Snapshot the row now: the source CPU block may be reused/clobbered
            # once persist returns, and the write runs later on a worker thread.
            # Force a copy — ``ascontiguousarray`` returns the input unchanged
            # when already contiguous, which would alias the caller's buffer.
            # The write then consumes this snapshot's byte-view zero-copy.
            row_copy = np.array(row, dtype=self._dtype, copy=True, order="C")
            with self._cache_lock:
                self._pending[key] = row_copy
            tasks.append(functools.partial(self._write_one, key, path, row_copy))
        if tasks:
            self._pool.enqueue_store(self._next_job_id(), len(tasks), tasks)

    def prefetch(self, keys: Sequence[bytes]) -> None:
        self._reap_finished()
        tasks = []
        for key in keys:
            # Skip keys already served from memory (write in flight or already
            # prefetched) — restore would hit those without touching disk.
            with self._cache_lock:
                if key in self._pending or key in self._prefetched:
                    continue
            tasks.append(functools.partial(self._read_one, key, self._path(key)))
        if tasks:
            self._pool.enqueue_load(self._next_job_id(), len(tasks), tasks)

    def restore(self, keys: Sequence[bytes]) -> np.ndarray | None:
        self._reap_finished()
        rows = np.empty((len(keys), *self._row_shape), dtype=self._dtype)
        for i, key in enumerate(keys):
            # Lookup order: pending write (read-after-write) -> prefetched
            # read-ahead -> synchronous disk read (fallback). Prefetched
            # entries are consumed: after restore the row lives in the CPU
            # primary tier, so the cache copy is no longer needed.
            with self._cache_lock:
                cached = self._pending.get(key)
                if cached is None:
                    cached = self._prefetched.pop(key, None)
            if cached is not None:
                rows[i] = cached
                continue
            row = self._read_row(self._path(key))
            if row is None:
                return None
            rows[i] = row
        return rows

    def shutdown(self, drain_timeout: float = 30.0) -> None:
        """Flush pending writes and stop the pool. Best-effort, idempotent.

        ``DualQueueThreadPool.shutdown`` discards still-queued tasks, so we
        first wait (bounded by ``drain_timeout``) for the pending writes to
        empty — each completed ``_write_one`` pops its key — ensuring queued
        sidecars actually reach disk before the pool stops. In-flight prefetch
        reads need no draining: they only warm a cache and are safe to drop.
        """
        if self._closed:
            return
        self._closed = True

        waited, step = 0.0, 0.005
        while waited < drain_timeout:
            with self._cache_lock:
                if not self._pending:
                    break
            self._pool.get_finished()
            time.sleep(step)
            waited += step

        with self._cache_lock:
            leaked = len(self._pending)
        if leaked:
            logger.warning(
                "routed-experts store shutdown: %d sidecar write(s) did not "
                "drain within %.1fs; they may be missing on disk",
                leaked,
                drain_timeout,
            )
        self._pool.shutdown(wait=True)


def _fs_routed_experts_root(ctx: RoutedExpertsStoreContext) -> str:
    """Resolve the filesystem root for fs-backed routing sidecars.

    Sit beside the KV block files under the tier's ``root_dir`` when set, so
    routing files share the KV layout; otherwise fall back to a stable temp
    subdir keyed by the instance id.
    """
    import tempfile

    root_dir = ctx.tier_config.get("root_dir")
    if root_dir:
        return os.path.join(str(root_dir), "routed_experts")
    instance_id = ctx.offloading_spec.vllm_config.instance_id
    return os.path.join(tempfile.gettempdir(), f"vllm_routed_experts_{instance_id}")


def build_fs_routed_experts_store(
    ctx: RoutedExpertsStoreContext,
) -> RoutedExpertsSecondaryStore:
    """Builder for the built-in filesystem secondary tier (``type="fs"``)."""
    from vllm.v1.kv_offload.file_mapper import FileMapper

    spec = ctx.offloading_spec
    root = _fs_routed_experts_root(ctx)
    file_mapper = FileMapper.from_offloading_spec(
        root_dir=root,
        offloading_spec=spec,
        gpu_blocks_per_file=spec.block_size_factor,
    )
    return FileRoutedExpertsStore(
        root_dir=root,
        file_mapper=file_mapper,
        row_shape=ctx.row_shape,
        dtype=ctx.dtype,
        # Sidecars are far smaller than KV blocks, so they need less IO
        # parallelism than the KV tier. A small pool avoids adding
        # GIL-contending threads to the EngineCore (scheduler) process.
        n_write_threads=int(ctx.tier_config.get("n_write_threads", 4)),
        n_read_threads=int(ctx.tier_config.get("n_read_threads", 4)),
    )


RoutedExpertsStoreFactory.register_store(
    "fs",
    "vllm.model_executor.layers.fused_moe.routed_experts_capture.store",
    "build_fs_routed_experts_store",
)


class _OffloadBuffer(Protocol):
    """The subset of RoutedExpertsManager the observer needs.

    Declared structurally so this module does not import the manager at
    runtime (the observer only reads/writes whole offloaded-block rows).
    """

    def read_cpu_blocks(self, cpu_block_ids: np.ndarray) -> np.ndarray: ...

    def write_cpu_blocks(self, cpu_block_ids: np.ndarray, rows: np.ndarray) -> None: ...


class RoutedExpertsBlockLifecycleObserver(BlockLifecycleObserver):
    """Mirror cascade / promotion events into a secondary routing store.

    Registered on a ``TieringOffloadingManager``. On cascade (CPU primary ->
    secondary) it persists the offloaded-block rows of the affected CPU
    blocks; on promotion (secondary -> CPU primary) it restores them, so the
    routed-experts offload buffer's lifecycle matches the KV cache's across
    all tiers (GPU <-> CPU <-> disk/object).
    """

    def __init__(
        self,
        manager: _OffloadBuffer,
        store: RoutedExpertsSecondaryStore,
    ) -> None:
        self._manager = manager
        self._store = store
        # Cumulative counters (blocks), for observability. The disk round-trip
        # is otherwise invisible; these let tests/ops confirm the routing
        # offload buffer actually followed KV through the secondary tier.
        self.cascaded_blocks = 0
        self.promoted_blocks = 0

    def on_blocks_cascaded(
        self, keys: Sequence[bytes], cpu_block_ids: np.ndarray
    ) -> None:
        if len(keys) == 0:
            return
        rows = self._manager.read_cpu_blocks(np.asarray(cpu_block_ids))
        self._store.persist(keys, rows)
        self.cascaded_blocks += len(keys)
        logger.debug(
            "routed-experts offload: cascaded %d block(s) to secondary (total=%d)",
            len(keys),
            self.cascaded_blocks,
        )

    def on_blocks_promotion_started(
        self, keys: Sequence[bytes], cpu_block_ids: np.ndarray
    ) -> None:
        # KV bytes just started loading secondary -> primary; warm the routing
        # read-ahead cache in parallel so the matching ``on_blocks_promoted``
        # restore (after the KV load completes) serves from memory.
        if len(keys) == 0:
            return
        self._store.prefetch(keys)

    def on_blocks_promoted(
        self, keys: Sequence[bytes], cpu_block_ids: np.ndarray
    ) -> None:
        if len(keys) == 0:
            return
        rows = self._store.restore(keys)
        if rows is None:
            # Missing sidecar: leave offloaded-block rows as-is. Should not
            # happen for a block the connector decided to promote, but never
            # corrupt the offload buffer with partial data.
            logger.warning(
                "routed-experts sidecar missing for %d promoted block(s); "
                "offloaded-block rows left unchanged",
                len(keys),
            )
            return
        self._manager.write_cpu_blocks(np.asarray(cpu_block_ids), rows)
        self.promoted_blocks += len(keys)
        logger.debug(
            "routed-experts offload: promoted %d block(s) from secondary (total=%d)",
            len(keys),
            self.promoted_blocks,
        )

    def shutdown(self) -> None:
        """Flush the secondary store's pending writes, if it has any."""
        store_shutdown = getattr(self._store, "shutdown", None)
        if callable(store_shutdown):
            store_shutdown()
