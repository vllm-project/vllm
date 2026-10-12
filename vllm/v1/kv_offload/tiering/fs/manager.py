# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FileSystemTierManager: pure-Python filesystem tier for KV cache offloading.

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
import mmap
import os
from collections.abc import Iterable
from itertools import groupby
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.backpressure import BackpressureDetector
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    _validate_offsets,
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
    """Pure-Python disk-backed secondary tier.

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
        backpressure_detector: BackpressureDetector | None = None,
        compact_groups: bool = False,
    ):
        """Args:
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
        backpressure_detector: Optional backpressure detector.
        compact_groups: Pack unequal BLHNC groups to reduce file size at the
            cost of additional CPU copies.

        """
        super().__init__(
            offloading_spec, primary_kv_view, tier_type, backpressure_detector
        )
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
        self._job_file_sizes: dict[JobId, list[int]] = {}
        # Per load job: how many blocks loaded before a failure (partial keep).
        # Written by the pool worker inside the load task before it raises (so
        # before task_done publishes the job); read on the scheduler thread in
        # get_finished_jobs only for job ids the finished queue returned. Under
        # the GIL that read cannot observe the finished job without the prior
        # write, so no extra lock is needed (get_finished is itself lock-free).
        self._load_progress: dict[JobId, int] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]
        config = offloading_spec.config
        self._packed_block_size = config.worker_kv_bytes_per_block
        self._group_sizes: dict[int, int] = {}
        storage_format = None
        if (
            compact_groups
            and config.groups
            and all(group.packed_layout for group in config.groups)
        ):
            group_sizes = {
                group.group_id: sum(size for _, size in group.packed_layout)
                for group in config.groups
            }
            if min(group_sizes.values()) < self._packed_block_size:
                self._group_sizes = group_sizes
                copies = (
                    1
                    if offloading_spec.replicated_layout
                    else config.parallel.world_size
                )
                self._packed_blocks = config.cache.blocks_per_chunk * copies
                assert self._block_size == round_up(
                    self._packed_blocks * self._packed_block_size, 4096
                )
                storage_format = {
                    "name": "packed-group-v1",
                    "layout": config.kv_cache_layout,
                    "block_bytes": self._packed_block_size,
                    "copies": copies,
                    "groups": [
                        (group.group_id, group.packed_layout) for group in config.groups
                    ],
                }

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=storage_format is None,
            storage_format=storage_format,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

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

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        keys = list(job_metadata.keys)
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = keys
        task = functools.partial(
            self._transfer_blocks,
            [self.file_mapper.get_file_name(key) for key in keys],
            [int(cid) * self._block_size for cid in job_metadata.chunk_ids],
            keys,
            False,
        )
        self._job_file_sizes[job_metadata.job_id] = [self._file_size(k) for k in keys]
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        # Track this load's keys so a failed promotion can mark only its failed
        # keys as a miss (see get_finished_jobs).
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        self._job_file_sizes[job_id] = [self._file_size(k) for k in keys]
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        offsets = [int(cid) * self._block_size for cid in job_metadata.chunk_ids]

        def load_task() -> None:
            try:
                self._transfer_blocks(paths, offsets, keys, True)
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

    def _file_size(self, key: OffloadKey) -> int:
        if not self._group_sizes:
            return self._block_size
        return round_up(
            self._group_sizes[get_offload_group_idx(key)] * self._packed_blocks, 4096
        )

    def _transfer_blocks(
        self, paths: list[str], offsets: list[int], keys: list[OffloadKey], load: bool
    ) -> None:
        transfer = batch_load_block if load else batch_store_block
        if not self._group_sizes:
            transfer(
                paths,
                self._primary_kv_view,
                offsets,
                self._block_size,
                self._use_o_direct,
            )
            return

        primary = self._primary_kv_view.cast("B")
        _validate_offsets(primary, offsets, self._block_size)
        sizes = [self._group_sizes[get_offload_group_idx(key)] for key in keys]
        completed = 0
        # Keep original key order: a failed load reports a successful prefix.
        for size, run in groupby(zip(paths, offsets, sizes), key=lambda item: item[2]):
            entries = list(run)
            if size == self._packed_block_size:
                try:
                    transfer(
                        [p for p, _, _ in entries],
                        primary,
                        [o for _, o, _ in entries],
                        self._block_size,
                        self._use_o_direct,
                    )
                except OSError as exc:
                    exc.num_succeeded = completed + getattr(exc, "num_succeeded", 0)  # type: ignore[attr-defined]
                    raise
                completed += len(entries)
                continue

            file_size = round_up(size * self._packed_blocks, 4096)
            # Bound staging memory to 16 MiB per task, or one larger file.
            batch_size = max(1, (16 * 1024 * 1024) // file_size)
            for start in range(0, len(entries), batch_size):
                batch = entries[start : start + batch_size]
                error = None
                with (
                    mmap.mmap(-1, len(batch) * file_size) as buffer,
                    memoryview(buffer) as packed,
                ):
                    if not load:
                        for i, (_, offset, _) in enumerate(batch):
                            for block in range(self._packed_blocks):
                                src = offset + block * self._packed_block_size
                                dst = i * file_size + block * size
                                packed[dst : dst + size] = primary[src : src + size]
                    succeeded = 0
                    try:
                        transfer(
                            [p for p, _, _ in batch],
                            packed,
                            [i * file_size for i in range(len(batch))],
                            file_size,
                            self._use_o_direct,
                        )
                        succeeded = len(batch)
                    except OSError as exc:
                        succeeded = getattr(exc, "num_succeeded", 0)
                        exc.num_succeeded = completed + succeeded  # type: ignore[attr-defined]
                        # I/O tracebacks retain memoryviews of the staging mmap.
                        error = exc.with_traceback(None)
                    if load:
                        for i, (_, offset, _) in enumerate(batch[:succeeded]):
                            for block in range(self._packed_blocks):
                                src = i * file_size + block * size
                                dst = offset + block * self._packed_block_size
                                primary[dst : dst + size] = packed[src : src + size]
                if error is not None:
                    raise error
                completed += len(batch)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Collect finished jobs; a failed promotion marks only its failed keys
        as a miss here (scheduler thread)."""
        results = []
        for job_id, success, transfer_time in self._pool.get_finished():
            file_sizes = self._job_file_sizes.pop(job_id, [])
            transfer_bytes = sum(file_sizes) if file_sizes else None
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
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
                        transfer_bytes=sum(file_sizes[:num_succeeded]),
                    )
                )
                continue
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    transfer_time=transfer_time,
                    transfer_bytes=transfer_bytes,
                )
            )
        return results

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
        """Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
