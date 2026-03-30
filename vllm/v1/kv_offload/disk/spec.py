# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieredOffloadingSpec: GPU → CPU (pinned) → Disk (NVMe).

Extends CPUOffloadingSpec with a disk tier. The GPU↔CPU path uses
pinned memory (fast). The CPU↔Disk path uses background threaded I/O.
"""
from collections.abc import Iterator

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.disk.manager import TieredOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCaches, OffloadingSpec
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)


class TieredOffloadingSpec(OffloadingSpec):

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise ValueError(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        self.disk_path: str = self.extra_config.get("disk_path", "")
        if not self.disk_path:
            raise ValueError(
                "disk_path must be specified for TieredOffloadingSpec"
            )

        disk_bytes = self.extra_config.get("disk_bytes_to_use")
        self.write_threshold = int(
            self.extra_config.get("write_threshold", 1)
        )
        self.io_threads = int(self.extra_config.get("io_threads", 4))

        # Compute block counts using same formula as CPUOffloadingSpec
        assert kv_cache_config is not None
        assert kv_cache_config.num_blocks > 0
        total_kv_bytes_per_worker = sum(
            t.size for t in kv_cache_config.kv_cache_tensors
        )
        page_size_bytes = total_kv_bytes_per_worker // kv_cache_config.num_blocks
        world_size = vllm_config.parallel_config.world_size

        kv_bytes_per_block = page_size_bytes * world_size
        kv_bytes_per_offloaded_block = (
            kv_bytes_per_block * self.block_size_factor
        )

        self.num_cpu_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )

        # Disk blocks: if disk_bytes specified, compute; otherwise default
        # to 3x CPU blocks (NVMe is cheap)
        if disk_bytes:
            self.num_disk_blocks = (
                int(disk_bytes) // kv_bytes_per_offloaded_block
                if kv_bytes_per_offloaded_block > 0
                else 0
            )
        else:
            self.num_disk_blocks = self.num_cpu_blocks * 3

        self.eviction_policy: str = self.extra_config.get(
            "eviction_policy", "lru"
        )

        # Scheduler-side
        self._manager: TieredOffloadingManager | None = None
        # Worker-side
        self._handlers: CpuGpuOffloadingHandlers | None = None

        logger.info(
            "TieredOffloadingSpec: cpu=%d blocks, disk=%d blocks, "
            "write_threshold=%d, io_threads=%d, disk_path=%s",
            self.num_cpu_blocks, self.num_disk_blocks,
            self.write_threshold, self.io_threads, self.disk_path,
        )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )

            assert len(self.gpu_block_size) == 1
            offloaded_block_size = (
                self.gpu_block_size[0] * self.block_size_factor
            )

            self._manager = TieredOffloadingManager(
                block_size=offloaded_block_size,
                num_cpu_blocks=self.num_cpu_blocks,
                num_disk_blocks=self.num_disk_blocks,
                write_threshold=self.write_threshold,
                cache_policy=self.eviction_policy,
                enable_events=enable_events,
            )
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]:
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise ValueError(
                    "Tiered offloading requires CUDA-alike GPUs"
                )

            # GPU↔CPU handlers (pinned memory, fast)
            self._handlers = CpuGpuOffloadingHandlers(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.num_cpu_blocks,
            )

            # Stash CPU tensors for the disk worker to access
            self._cpu_tensors = [
                t for t in self._get_cpu_tensors()
            ]

        assert self._handlers is not None
        yield (
            GPULoadStoreSpec,
            CPULoadStoreSpec,
            self._handlers.gpu_to_cpu_handler,
        )
        yield (
            CPULoadStoreSpec,
            GPULoadStoreSpec,
            self._handlers.cpu_to_gpu_handler,
        )

    def _get_cpu_tensors(self) -> list:
        """Extract CPU tensors from the handler for disk I/O.
        CPU tensors are the src for CPU→GPU (load) and dst for GPU→CPU (store).
        """
        handler = self._handlers
        if handler is None:
            return []
        # cpu_to_gpu_handler: src=CPU, dst=GPU
        return handler.cpu_to_gpu_handler.src_tensors

    def get_disk_worker_config(self) -> dict:
        """Config for the worker to create a DiskIOWorker."""
        return {
            "disk_path": self.disk_path,
            "num_disk_blocks": self.num_disk_blocks,
            "io_threads": self.io_threads,
        }
