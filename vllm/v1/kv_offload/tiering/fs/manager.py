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
from collections.abc import Iterable
from typing import TYPE_CHECKING

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.common import FileSystemLoadStoreSpec
from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
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
        if not self._tier.uses_worker_transfers():
            paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
            return self._path_exists_batch(paths)

        paths_by_key = [self._tier.get_lookup_paths(k) for k in keys]
        flat_paths = [path for paths in paths_by_key for path in paths]
        flat_results = self._path_exists_batch(flat_paths)

        results = []
        offset = 0
        for paths in paths_by_key:
            next_offset = offset + len(paths)
            results.append(all(flat_results[offset:next_offset]))
            offset = next_offset
        return results

    @staticmethod
    def _path_exists_batch(paths: list[str]) -> list[bool]:
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            return batch_lookup_C(paths)
        return [os.path.exists(p) for p in paths]


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        worker_transfers: bool = False,
    ):
        """
        Args:
            offloading_spec: contains the vllm_config, kv_cache_config
                and block_size_factor.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            worker_transfers: If true, scheduler still owns policy/lookup but
                workers execute CPU-shard <-> filesystem data copies.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            gpu_blocks_per_file=offloading_spec.block_size_factor,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)
        self._worker_transfers = worker_transfers
        self._worker_store_paths: dict[int, list[tuple[str, str]]] = {}

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
    def uses_worker_transfers(self) -> bool:
        return self._worker_transfers

    def _get_num_ranks(self) -> int:
        num_ranks = self._offloading_spec.vllm_config.parallel_config.world_size
        if num_ranks <= 0:
            raise ValueError(f"Invalid parallel world size: {num_ranks}")
        return num_ranks

    def _get_file_paths(self, keys, rank=None) -> list[str]:
        return [self.file_mapper.get_file_name(k, rank=rank) for k in keys]

    def get_lookup_paths(self, key: OffloadKey) -> list[str]:
        if not self._worker_transfers:
            return [self.file_mapper.get_file_name(key)]
        return [
            self.file_mapper.get_file_name(key, rank=rank)
            for rank in range(self._get_num_ranks())
        ]

    def _get_temp_path(self, final_path: str, job_id: int) -> str:
        instance_id = self._offloading_spec.vllm_config.instance_id
        return f"{final_path}.{instance_id}.{job_id}.tmp"

    @override
    def build_worker_store_transfer(
        self, job_metadata: JobMetadata
    ) -> tuple[CPULoadStoreSpec, FileSystemLoadStoreSpec]:
        num_ranks = self._get_num_ranks()
        all_final: list[str] = []
        all_temp: list[str] = []
        for rank in range(num_ranks):
            fp = self._get_file_paths(job_metadata.keys, rank=rank)
            tp = [self._get_temp_path(p, job_metadata.job_id) for p in fp]
            key = job_metadata.job_id
            if key not in self._worker_store_paths:
                self._worker_store_paths[key] = []
            self._worker_store_paths[key].extend(zip(tp, fp))
            all_final.extend(fp)
            all_temp.extend(tp)
        logger.debug(
            "Built filesystem worker store transfer job %d: %d keys, %d ranks",
            job_metadata.job_id,
            len(job_metadata.keys),
            num_ranks,
        )
        return CPULoadStoreSpec([int(b) for b in job_metadata.block_ids]), (
            FileSystemLoadStoreSpec(
                file_paths=all_final,
                temp_file_paths=all_temp,
                block_size=self._block_size,
                num_ranks=num_ranks,
            )
        )

    @override
    def build_worker_load_transfer(
        self, job_metadata: JobMetadata
    ) -> tuple[FileSystemLoadStoreSpec, CPULoadStoreSpec]:
        num_ranks = self._get_num_ranks()
        all_final: list[str] = []
        for rank in range(num_ranks):
            fp = self._get_file_paths(job_metadata.keys, rank=rank)
            all_final.extend(fp)
        logger.debug(
            "Built filesystem worker load transfer job %d: %d keys, %d ranks",
            job_metadata.job_id,
            len(job_metadata.keys),
            num_ranks,
        )
        return (
            FileSystemLoadStoreSpec(
                file_paths=all_final,
                block_size=self._block_size,
                num_ranks=num_ranks,
            ),
            CPULoadStoreSpec([int(b) for b in job_metadata.block_ids]),
        )

    @override
    def complete_worker_store(
        self, job_metadata: JobMetadata, success: bool
    ) -> None:
        paths = self._worker_store_paths.pop(job_metadata.job_id, [])
        if not success:
            for temp_path, _ in paths:
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass
            return

        missing_temp_paths = [
            temp_path
            for temp_path, final_path in paths
            if not os.path.exists(final_path) and not os.path.exists(temp_path)
        ]
        if missing_temp_paths:
            for temp_path, _ in paths:
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass
            logger.warning(
                "Discarding incomplete worker filesystem store job %d: "
                "missing %d temp file(s). Verify that root_dir is visible "
                "from every worker when worker_transfers is enabled.",
                job_metadata.job_id,
                len(missing_temp_paths),
            )
            return

        for temp_path, final_path in paths:
            if os.path.exists(final_path):
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass
                continue
            os.replace(temp_path, final_path)

    @override
    def abort_worker_transfers(self) -> None:
        paths_by_job = self._worker_store_paths
        self._worker_store_paths = {}
        for paths in paths_by_job.values():
            for temp_path, _ in paths:
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        tasks = (
            functools.partial(
                store_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        tasks = (
            functools.partial(
                load_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        return (
            JobResult(job_id=job_id, success=success)
            for job_id, success in self._pool.get_finished()
        )

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
