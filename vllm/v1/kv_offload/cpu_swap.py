# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPUSwapSpec: OffloadingSpec for CPU swap mode.

Like CPUOffloadingSpec but asserts that the CPU has enough memory to hold
all KV cache blocks (no eviction). Uses SwapManager instead of LRU/ARC.
"""
from collections.abc import Iterator

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.swap_manager import SwapManager
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)


class CPUSwapSpec(OffloadingSpec):
    """
    Spec for CPU swap mode where the entire KV cache is swapped
    between CPU and GPU on a per-layer basis.

    Asserts at initialization that the CPU has enough memory to hold
    all possible KV cache blocks.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # Calculate kv_bytes_per_offloaded_block
        assert kv_cache_config is not None
        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )

        self.num_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )

        # Calculate the total number of GPU blocks that could be allocated.
        # The CPU must be able to hold all of them.
        total_gpu_blocks = sum(
            kv_cache_tensor.size // page_size_bytes
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors
        )
        # Convert GPU blocks to offloaded block units
        block_size_factor = self.offloaded_block_size // self.gpu_block_size
        required_offloaded_blocks = (
            total_gpu_blocks + block_size_factor - 1
        ) // block_size_factor

        assert self.num_blocks >= required_offloaded_blocks, (
            f"CPU swap mode requires enough CPU memory to hold all KV cache. "
            f"CPU can hold {self.num_blocks} offloaded blocks but "
            f"{required_offloaded_blocks} are needed. "
            f"Increase cpu_bytes_to_use to at least "
            f"{required_offloaded_blocks * kv_bytes_per_offloaded_block} bytes."
        )

        logger.info(
            "CPU swap spec: %d CPU blocks available, %d required "
            "(%.1f%% utilization)",
            self.num_blocks,
            required_offloaded_blocks,
            100.0 * required_offloaded_blocks / self.num_blocks
            if self.num_blocks > 0
            else 0,
        )

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handlers: CpuGpuOffloadingHandlers | None = None

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )

            backend = CPUBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_blocks,
            )

            self._manager = SwapManager(
                backend=backend, enable_events=enable_events
            )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]:
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "CPU Swap is currently only supported on CUDA-alike GPUs"
                )

            self._handlers = CpuGpuOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=self.num_blocks,
                gpu_caches=kv_caches,
            )

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler
