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

File naming:  <base_path>/<hhh>/<hh>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
from vllm.v1.kv_offload.tiering.fs.io import store_block, load_block
from vllm.v1.kv_offload.tiering.fs.file_mapper import FileMapper

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished() polls job completion and returns completed JobResults.

    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        primary_kv_view: memoryview,
        root_dir: str,
        kv_cache_config: "KVCacheConfig",
        gpu_blocks_per_file: int = 1,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
    ):
        """
        Args:
            vllm_config: Global vLLM configuration.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            root_dir: Root directory for block files.
            kv_cache_config: KV cache configuration.
            gpu_blocks_per_file: Number of GPU blocks per file.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
        """
        super().__init__(vllm_config, primary_kv_view)
        
        # Extract block size from primary view
        assert primary_kv_view.strides is not None, "primary_kv_view.strides cannot be None"
        self._block_size: int = primary_kv_view.strides[0]
        
        # Create file mapper
        self.file_mapper = FileMapper.from_vllm_config(
            root_dir=root_dir,
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            gpu_blocks_per_file=gpu_blocks_per_file,
        )
        self.file_mapper.write_run_config()
        
        # Create thread pool with file mapper
        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            self.file_mapper,
            thread_name_prefix="vllm_kv_py_fs",
        )

    def lookup(self, key: OffloadKey, req_context: ReqContext | None = None) -> bool | None:
        return os.path.exists(self.file_mapper.get_file_name(key))

    def submit_store(self, job_metadata: JobMetadata) -> None:
        self._pool.enqueue_store(
            job_metadata.job_id, job_metadata.keys, job_metadata.block_ids,
            self._primary_kv_view, self._block_size, store_block,
        )

    def submit_load(self, job_metadata: JobMetadata) -> None:
        self._pool.enqueue_load(
            job_metadata.job_id, job_metadata.keys, job_metadata.block_ids,
            self._primary_kv_view, self._block_size, load_block,
        )

    def get_finished(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        return (
            JobResult(job_id=job_id, success=success)
            for job_id, success in self._pool.get_finished()
        )

    @staticmethod
    def get_tier_type() -> str:
        return "fs_python"
