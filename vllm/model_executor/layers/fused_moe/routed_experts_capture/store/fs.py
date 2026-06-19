# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filesystem secondary tier: one ``.re`` sidecar file per offloaded block.

The built-in ``type="fs"`` backend. Writes are asynchronous (a thread pool
keeps the scheduler off the disk-IO critical path) and reads are prefetched,
mirroring the KV ``FileSystemTierManager``; routing sidecars sit beside the KV
block files via the same ``FileMapper`` layout. Registers itself with
``RoutedExpertsStoreFactory`` under ``"fs"`` at import time.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from collections.abc import Sequence

import numpy as np

from vllm.model_executor.layers.fused_moe.routed_experts_capture.store.base import (
    RoutedExpertsSecondaryStore,
    RoutedExpertsStoreContext,
    RoutedExpertsStoreFactory,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

logger = logging.getLogger(__name__)


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
    "vllm.model_executor.layers.fused_moe.routed_experts_capture.store.fs",
    "build_fs_routed_experts_store",
)
