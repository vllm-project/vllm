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
    get_offload_block_hash,
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
        paths = [self._tier.get_file_name(key) for key in keys]
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
        path_sharding: str | None = None,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files, or an ordered,
                comma-separated list of roots when path sharding is enabled.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
            path_sharding: Set to ``"by_block_hash"`` to map each complete
                logical block to one of the configured roots using its stable
                content hash.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
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
        # Per load job: whether each key completed before a task failed. This
        # supports independent per-root batches whose successful keys need not
        # form one prefix in the original job order.
        self._load_success: dict[JobId, list[bool]] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        raw_root_dirs = root_dir.split(",")
        if not raw_root_dirs or any(not path.strip() for path in raw_root_dirs):
            raise ValueError("root_dir must contain non-empty filesystem paths")
        self._root_dirs = [path.strip() for path in raw_root_dirs]
        if path_sharding not in (None, "by_block_hash"):
            raise ValueError(
                "path_sharding must be omitted or set to 'by_block_hash', got "
                f"{path_sharding!r}"
            )
        if len(self._root_dirs) > 1 and path_sharding != "by_block_hash":
            raise ValueError(
                "multiple root_dir paths require path_sharding='by_block_hash'"
            )
        normalized_roots = [os.path.normpath(path) for path in self._root_dirs]
        if len(set(normalized_roots)) != len(normalized_roots):
            raise ValueError("root_dir paths must be distinct")

        # Every mapper describes the existing complete-row block format. Only
        # the root differs; the hash chooses exactly one mapper for each key.
        self.file_mappers = [
            FileMapper.from_offloading_spec(
                root_dir=path,
                offloading_spec=offloading_spec,
                blocks_per_file=offloading_spec.blocks_per_chunk,
                parallel_agnostic=True,
            )
            for path in self._root_dirs
        ]
        # Preserve the historical attribute for single-root users and tools.
        self.file_mapper = self.file_mappers[0]

        # Write the same full-row format config under every root.
        config_paths = []
        for mapper in self.file_mappers:
            config_path = mapper.get_config_file_path()
            config_paths.append(config_path)
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    json.dump(mapper.get_run_config(), f, indent=2, sort_keys=True)

        # Detect direct-I/O support and alignment independently for each root.
        # The I/O layer also checks every transfer's size and buffer address;
        # only an ineligible operation falls back to buffered I/O.
        self._o_direct_supported: list[bool] = []
        self._direct_io_alignments: list[int] = []
        self._filesystem_block_sizes: list[int] = []
        for root, config_path in zip(self._root_dirs, config_paths):
            directory = os.path.dirname(config_path)
            supported = probe_o_direct(directory)
            try:
                filesystem_block_size = os.statvfs(directory).f_bsize
            except OSError as exc:
                logger.warning(
                    "Could not query filesystem block size for '%s': %s.",
                    root,
                    exc,
                )
                filesystem_block_size = 0
            # probe_o_direct performs a page-sized transfer from a page-aligned
            # mmap. Its success proves page alignment is valid. statvfs.f_bsize
            # is informational here: NFS commonly reports its 1 MiB preferred
            # transfer size rather than the kernel's strict O_DIRECT alignment.
            direct_io_alignment = mmap.PAGESIZE if supported else 0
            self._o_direct_supported.append(supported)
            self._direct_io_alignments.append(direct_io_alignment)
            self._filesystem_block_sizes.append(filesystem_block_size)
            if not supported:
                logger.warning(
                    "O_DIRECT is unavailable at '%s'; operations on this "
                    "root will use buffered I/O.",
                    root,
                )
            else:
                logger.info(
                    "O_DIRECT enabled at '%s' with validated alignment %d "
                    "bytes (statvfs block size %d bytes).",
                    root,
                    direct_io_alignment,
                    filesystem_block_size,
                )

        # Preserve the historical aggregate attribute for callers and tests.
        self._use_o_direct = all(self._o_direct_supported)

        if len(self.file_mappers) > 1:
            logger.info(
                "Configured whole-block hash sharding across %d FS roots",
                len(self.file_mappers),
            )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    def _get_mapper_index(self, key: OffloadKey) -> int:
        block_hash = get_offload_block_hash(key)
        return int.from_bytes(block_hash, byteorder="big") % len(self.file_mappers)

    def get_file_name(self, key: OffloadKey) -> str:
        """Map a complete logical block to its deterministic storage root."""
        return self.file_mappers[self._get_mapper_index(key)].get_file_name(key)

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

        paths_by_root: list[list[str]] = [[] for _ in self.file_mappers]
        offsets_by_root: list[list[int]] = [[] for _ in self.file_mappers]
        for key, block_id in zip(keys, job_metadata.block_ids):
            root_idx = self._get_mapper_index(key)
            paths_by_root[root_idx].append(
                self.file_mappers[root_idx].get_file_name(key)
            )
            offsets_by_root[root_idx].append(int(block_id) * self._block_size)

        tasks = [
            functools.partial(
                batch_store_block,
                paths,
                self._primary_kv_view,
                offsets,
                self._block_size,
                o_direct_supported,
                direct_io_alignment,
            )
            for paths, offsets, o_direct_supported, direct_io_alignment in zip(
                paths_by_root,
                offsets_by_root,
                self._o_direct_supported,
                self._direct_io_alignments,
            )
        ]
        self._pool.enqueue_store(job_metadata.job_id, len(tasks), tasks)

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        # Track this load's keys so a failed promotion can mark only its failed
        # keys as a miss (see get_finished_jobs).
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        self._load_success[job_id] = [False] * len(keys)
        paths_by_root: list[list[str]] = [[] for _ in self.file_mappers]
        offsets_by_root: list[list[int]] = [[] for _ in self.file_mappers]
        indices_by_root: list[list[int]] = [[] for _ in self.file_mappers]
        for key_idx, (key, block_id) in enumerate(zip(keys, job_metadata.block_ids)):
            root_idx = self._get_mapper_index(key)
            paths_by_root[root_idx].append(
                self.file_mappers[root_idx].get_file_name(key)
            )
            offsets_by_root[root_idx].append(int(block_id) * self._block_size)
            indices_by_root[root_idx].append(key_idx)

        tasks = []
        for root_idx, (
            paths,
            offsets,
            key_indices,
            o_direct_supported,
            direct_io_alignment,
        ) in enumerate(
            zip(
                paths_by_root,
                offsets_by_root,
                indices_by_root,
                self._o_direct_supported,
                self._direct_io_alignments,
            )
        ):

            def load_task(
                root_idx: int = root_idx,
                paths: list[str] = paths,
                offsets: list[int] = offsets,
                key_indices: list[int] = key_indices,
                o_direct_supported: bool = o_direct_supported,
                direct_io_alignment: int = direct_io_alignment,
            ) -> None:
                try:
                    batch_load_block(
                        paths,
                        self._primary_kv_view,
                        offsets,
                        self._block_size,
                        o_direct_supported,
                        direct_io_alignment,
                    )
                except Exception as exc:
                    num_succeeded = getattr(exc, "num_succeeded", 0)
                    for key_idx in key_indices[:num_succeeded]:
                        self._load_success[job_id][key_idx] = True
                    logger.debug(
                        "Load of %d blocks for job %s root %d failed at "
                        "root-local block %d: %s",
                        len(paths),
                        job_id,
                        root_idx,
                        num_succeeded,
                        exc,
                    )
                    raise
                else:
                    for key_idx in key_indices:
                        self._load_success[job_id][key_idx] = True

            tasks.append(load_task)

        self._pool.enqueue_load(job_id, len(tasks), tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Collect finished jobs; a failed promotion marks only its failed keys
        as a miss here (scheduler thread)."""
        results = []
        for job_id, success, transfer_time in self._pool.get_finished():
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
