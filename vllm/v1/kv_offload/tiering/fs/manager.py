# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            return batch_lookup_C(paths)
        return (os.path.exists(p) for p in paths)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        KV cache sharing between multiple vLLM instances using the same
        ``root_dir`` (e.g., via a shared PVC) works by default: ``NONE_HASH``
        (the chain-hash seed for block content hashes) is derived from a fixed
        default seed, so identical token content produces identical block
        filenames across instances. Setting the ``PYTHONHASHSEED`` environment
        variable to the same value on all instances overrides the default seed,
        and is required to share a cache when using a non-cryptographic
        prefix-caching hash algorithm, which seeds ``NONE_HASH`` randomly.
    """

    medium: ClassVar[Medium] = Medium.STORAGE
    CACHE_BYTES = "vllm:kv_offload_fs_cache_bytes"
    CACHE_ENTRIES = "vllm:kv_offload_fs_cache_entries"

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        del extra_config
        return {
            cls.CACHE_BYTES: OffloadingGaugeMetadata(
                documentation="Current filesystem KV cache block-data bytes."
            ),
            cls.CACHE_ENTRIES: OffloadingGaugeMetadata(
                documentation="Current filesystem KV cache block entry count."
            ),
        }

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
        max_bytes: int | None = None,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
            max_bytes: Maximum block-data bytes stored by this filesystem tier.
                ``None`` preserves the historical unbounded behavior.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        if isinstance(max_bytes, bool) or (
            max_bytes is not None and not isinstance(max_bytes, int)
        ):
            raise TypeError("max_bytes must be a non-negative integer or None")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer or None")
        self.max_bytes = max_bytes
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Keys of in-flight load (promotion) jobs, so a failed load can mark
        # its own cached lookup verdicts False (see get_finished_jobs).
        self._load_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Per load job: how many blocks loaded before a failure (partial keep).
        # Written by the pool worker inside the load task before it raises (so
        # before task_done publishes the job); read on the scheduler thread in
        # get_finished_jobs only for job ids the finished queue returned. Under
        # the GIL that read cannot observe the finished job without the prior
        # write, so no extra lock is needed (get_finished is itself lock-free).
        self._load_progress: dict[JobId, int] = {}
        self._load_paths: dict[JobId, list[str]] = {}
        self._skipped_store_jobs: set[JobId] = set()
        self._evicted_store_keys: dict[JobId, list[OffloadKey]] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )
        self._storage_dir = f"{self.file_mapper.base_path}_r{self.file_mapper.rank}"
        self._capacity_lock = threading.Lock()
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._cache_bytes = 0
        self._reserved_bytes = 0
        self._protected_paths: set[str] = set()

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        if max_bytes is not None:
            self._scan_cache()
            with self._capacity_lock:
                self._evict_locked(max(self._cache_bytes - max_bytes, 0), set())

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # rather than failing every block.
        self._use_o_direct = probe_o_direct(os.path.dirname(config_path))
        if not self._use_o_direct:
            logger.warning(
                "O_DIRECT is not supported at '%s'; falling back to buffered "
                "I/O for the '%s' KV offload tier.",
                root_dir,
                tier_type,
            )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    def _scan_cache(self) -> None:
        """Build the LRU index once at startup from file modification times."""
        entries = []
        if os.path.isdir(self._storage_dir):
            for dirpath, _, filenames in os.walk(self._storage_dir):
                for filename in filenames:
                    if not filename.endswith(".bin"):
                        continue
                    path = os.path.join(dirpath, filename)
                    try:
                        stat = os.stat(path)
                    except OSError:
                        continue
                    entries.append((stat.st_mtime_ns, path, stat.st_size))
        entries.sort()
        self._entries.update((path, size) for _, path, size in entries)
        self._cache_bytes = sum(self._entries.values())

    @staticmethod
    def _path_key(path: str) -> OffloadKey | None:
        try:
            group_dir = os.path.basename(os.path.dirname(path))
            group_idx = int(group_dir.rsplit("_g", 1)[1])
            block_hash = bytes.fromhex(os.path.basename(path)[:-4])
            return make_offload_key(block_hash, group_idx)
        except (IndexError, ValueError):
            return None

    def _evict_locked(
        self, bytes_to_free: int, protected: set[str]
    ) -> tuple[list[OffloadKey], int]:
        if bytes_to_free <= 0:
            return [], 0
        evicted = []
        freed = 0
        candidates = []
        available = 0
        for path, size in self._entries.items():
            if path in protected or path in self._protected_paths:
                continue
            candidates.append((path, size))
            available += size
        if available < bytes_to_free:
            return [], 0
        for path, size in candidates:
            if freed >= bytes_to_free:
                break
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                continue
            self._entries.pop(path, None)
            self._cache_bytes -= size
            bytes_to_free -= size
            freed += size
            key = self._path_key(path)
            if key is not None:
                evicted.append(key)
        return evicted, freed

    def _store_batch(self, job_id: JobId, paths: list[str], offsets: list[int]) -> None:
        if self.max_bytes is None:
            batch_store_block(
                paths,
                self._primary_kv_view,
                offsets,
                self._block_size,
                self._use_o_direct,
            )
            return
        unique = dict(zip(paths, offsets))
        with self._capacity_lock:
            missing = []
            for path, offset in unique.items():
                if path in self._entries:
                    self._entries.move_to_end(path)
                elif os.path.exists(path):
                    try:
                        size = os.path.getsize(path)
                    except OSError:
                        missing.append((path, offset))
                    else:
                        self._entries[path] = size
                        self._cache_bytes += size
                else:
                    missing.append((path, offset))

            needed = len(missing) * self._block_size
            if self.max_bytes is not None:
                if needed > self.max_bytes:
                    self._skipped_store_jobs.add(job_id)
                    logger.warning(
                        "Skipping filesystem KV cache store for job %s: "
                        "the batch does not fit in max_bytes=%s.",
                        job_id,
                        self.max_bytes,
                    )
                    return
                required = max(
                    self._cache_bytes + self._reserved_bytes + needed - self.max_bytes,
                    0,
                )
                evicted, freed = self._evict_locked(required, set(unique))
                required -= freed
                if required > 0:
                    self._skipped_store_jobs.add(job_id)
                    logger.warning(
                        "Skipping filesystem KV cache store for job %s: "
                        "the batch does not fit in max_bytes=%s.",
                        job_id,
                        self.max_bytes,
                    )
                    return
                if evicted:
                    self._evicted_store_keys[job_id] = evicted
            if not missing:
                return
            store_paths = [path for path, _ in missing]
            self._reserved_bytes += needed
            self._protected_paths.update(store_paths)

        success = False
        try:
            batch_store_block(
                store_paths,
                self._primary_kv_view,
                [offset for _, offset in missing],
                self._block_size,
                self._use_o_direct,
            )
            success = True
        finally:
            with self._capacity_lock:
                self._reserved_bytes -= needed
                self._protected_paths.difference_update(store_paths)
                if success:
                    for path in store_paths:
                        try:
                            size = os.path.getsize(path)
                        except OSError:
                            continue
                        self._cache_bytes += size - self._entries.get(path, 0)
                        self._entries[path] = size
                        self._entries.move_to_end(path)

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        keys = list(job_metadata.keys)
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = keys
        task = functools.partial(
            self._store_batch,
            job_metadata.job_id,
            [self.file_mapper.get_file_name(key) for key in keys],
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        # Track this load's keys so a failed promotion can mark only its failed
        # keys as a miss (see get_finished_jobs).
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        if self.max_bytes is not None:
            self._load_paths[job_id] = paths
            with self._capacity_lock:
                self._protected_paths.update(paths)
        offsets = [int(bid) * self._block_size for bid in job_metadata.block_ids]

        def load_task() -> None:
            try:
                batch_load_block(
                    paths,
                    self._primary_kv_view,
                    offsets,
                    self._block_size,
                    self._use_o_direct,
                )
            except OSError as exc:
                # Runs on the pool worker thread. Record how many blocks loaded
                # before the failure so get_finished_jobs can keep them; this
                # write precedes task_done, so the scheduler reads it safely
                # under the GIL once the finished queue hands back this job.
                num_succeeded = getattr(exc, "num_succeeded", 0)
                self._load_progress[job_id] = num_succeeded
                # Surfaces errno (e.g. EMFILE "Too many open files") for both
                # the C and Python load paths.
                logger.debug(
                    "Load of %d blocks for job %s failed at block %d: %s",
                    len(paths),
                    job_id,
                    num_succeeded,
                    exc,
                )
                raise

        self._pool.enqueue_load(job_id, 1, [load_task])

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Collect finished jobs; a failed promotion marks only its failed keys
        as a miss here (scheduler thread)."""
        results = []
        for job_id, success, transfer_time in self._pool.get_finished():
            with self._capacity_lock:
                self._protected_paths.difference_update(
                    self._load_paths.pop(job_id, ())
                )
                skipped = job_id in self._skipped_store_jobs
                self._skipped_store_jobs.discard(job_id)
                evicted = self._evicted_store_keys.pop(job_id, ())
            success = success and not skipped
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if evicted:
                    self.events.append(
                        OffloadingEvent(
                            keys=evicted,
                            medium=self.medium,
                            removed=True,
                            locality=self.locality,
                        )
                    )
                if success and keys and not skipped:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            load_keys = self._load_job_keys.pop(job_id, None)
            num_succeeded = self._load_progress.pop(job_id, 0)
            if load_keys is not None and not success:
                # A batched load stops at the first bad block and reports how
                # many loaded before it. Those earlier blocks are kept in the
                # primary tier (reported via successful_keys); only this block
                # and the ones after it are marked a miss and recomputed.
                successful = load_keys[:num_succeeded]
                failed = load_keys[num_succeeded:]
                self._lookup_manager.mark_miss(failed)
                results.append(
                    JobResult(
                        job_id=job_id,
                        success=False,
                        successful_keys=tuple(successful) if successful else None,
                        transfer_time=transfer_time,
                    )
                )
                continue
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    transfer_time=transfer_time,
                )
            )
        return results

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        del req_context
        if self.max_bytes is None:
            return
        with self._capacity_lock:
            for key in keys:
                path = self.file_mapper.get_file_name(key)
                if path in self._entries:
                    self._entries.move_to_end(path)

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        if self.max_bytes is None:
            return None
        with self._capacity_lock:
            stats = OffloadingConnectorStats()
            stats.set_gauge(self.CACHE_BYTES, self._cache_bytes)
            stats.set_gauge(self.CACHE_ENTRIES, len(self._entries))
            return stats

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
