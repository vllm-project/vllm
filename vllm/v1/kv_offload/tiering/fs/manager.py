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
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper, ShardedFileMapper
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
        path_sharding: str | None = None,
    ):
        """Args:
        offloading_spec: Contains normalized offloading configuration and
            blocks_per_chunk.
        primary_kv_view: Memoryview of the primary tier's CPU KV cache.
        tier_type: Tier type identifier, set by SecondaryTierFactory.
        root_dir: Root directory for block files, or comma-separated roots.
        n_read_threads: Number of read-priority I/O threads.
        n_write_threads: Number of write-priority I/O threads.
        enable_kv_events: Emit BlockStored KV events for blocks
            successfully stored to this tier. Effective only when KV
            cache events are enabled globally (kv_events_config).
        locality: Whether this tier's storage is LOCAL or REMOTE relative
            to the publishing vLLM instance.
        backpressure_detector: Optional backpressure detector.
        path_sharding: Set to ``"by_block_hash"`` to shard blocks across
            multiple storage roots.

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
        # Block count per in-flight job, used to report transfer_bytes.
        self._job_block_counts: dict[JobId, int] = {}
        # Per load job: whether each key loaded successfully before a failure.
        self._load_success: dict[JobId, list[bool]] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        if path_sharding not in (None, "by_block_hash"):
            raise ValueError(
                "path_sharding must be omitted or set to 'by_block_hash', got "
                f"{path_sharding!r}"
            )
        if "," in root_dir and path_sharding != "by_block_hash":
            raise ValueError(
                "multiple root_dir paths require path_sharding='by_block_hash'"
            )

        if path_sharding == "by_block_hash" or "," in root_dir:
            self.file_mapper: FileMapper = ShardedFileMapper.from_offloading_spec(
                root_dir=root_dir,
                offloading_spec=offloading_spec,
                blocks_per_file=offloading_spec.blocks_per_chunk,
                parallel_agnostic=True,
                path_sharding="by_block_hash",
            )
        else:
            self.file_mapper = FileMapper.from_offloading_spec(
                root_dir=root_dir,
                offloading_spec=offloading_spec,
                blocks_per_file=offloading_spec.blocks_per_chunk,
                parallel_agnostic=True,
            )

        # Write config file(s)
        config_paths = self.file_mapper.get_config_file_paths()
        for config_path in config_paths:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    json.dump(
                        self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                    )

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # or when block size is not aligned to the system page size.
        o_direct_supported = all(
            probe_o_direct(os.path.dirname(cp)) for cp in config_paths
        )
        is_aligned = self._block_size % mmap.PAGESIZE == 0
        self._use_o_direct = o_direct_supported and is_aligned
        if not self._use_o_direct:
            if not o_direct_supported:
                logger.warning(
                    "O_DIRECT is not supported at '%s'; falling back to buffered "
                    "I/O for the '%s' KV offload tier.",
                    root_dir,
                    tier_type,
                )
            elif not is_aligned:
                logger.warning(
                    "Block size (%d) is not a multiple of page size (%d); "
                    "falling back to buffered I/O for the '%s' KV offload tier.",
                    self._block_size,
                    mmap.PAGESIZE,
                    tier_type,
                )

        if self.file_mapper.num_shards > 1:
            logger.info(
                "Configured whole-block hash sharding across %d FS roots",
                self.file_mapper.num_shards,
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
        shards = self.file_mapper.group_by_shard(
            keys, job_metadata.chunk_ids, self._block_size
        )
        tasks = [
            functools.partial(
                batch_store_block,
                paths,
                self._primary_kv_view,
                offsets,
                self._block_size,
                self._use_o_direct,
            )
            for paths, offsets, _ in shards
        ]
        self._job_block_counts[job_metadata.job_id] = len(keys)
        self._pool.enqueue_store(job_metadata.job_id, len(tasks), tasks)

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        self._load_success[job_id] = [False] * len(keys)
        self._job_block_counts[job_id] = len(keys)
        shards = self.file_mapper.group_by_shard(
            keys, job_metadata.chunk_ids, self._block_size
        )
        tasks = []
        for paths, offsets, indices in shards:

            def load_task(
                paths: list[str] = paths,
                offsets: list[int] = offsets,
                indices: list[int] = indices,
            ) -> None:
                try:
                    batch_load_block(
                        paths,
                        self._primary_kv_view,
                        offsets,
                        self._block_size,
                        self._use_o_direct,
                    )
                except OSError as exc:
                    num_succeeded = getattr(exc, "num_succeeded", 0)
                    for idx in indices[:num_succeeded]:
                        self._load_success[job_id][idx] = True
                    logger.debug(
                        "Load of %d blocks for job %s failed at block %d: %s",
                        len(paths),
                        job_id,
                        num_succeeded,
                        exc,
                    )
                    raise
                else:
                    for idx in indices:
                        self._load_success[job_id][idx] = True

            tasks.append(load_task)

        self._pool.enqueue_load(job_id, len(tasks), tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Collect finished jobs; a failed promotion marks only its failed keys
        as a miss here (scheduler thread)."""
        results = []
        for job_id, success, transfer_time in self._pool.get_finished():
            block_count = self._job_block_counts.pop(job_id, 0)
            transfer_bytes = block_count * self._block_size if block_count else None
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
            load_success = self._load_success.pop(job_id, None)
            if load_keys is not None and not success:
                assert load_success is not None
                successful = [
                    key for key, loaded in zip(load_keys, load_success) if loaded
                ]
                failed = [
                    key for key, loaded in zip(load_keys, load_success) if not loaded
                ]
                self._lookup_manager.mark_miss(failed)
                results.append(
                    JobResult(
                        job_id=job_id,
                        success=False,
                        successful_keys=tuple(successful) if successful else None,
                        transfer_time=transfer_time,
                        transfer_bytes=transfer_bytes,
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
