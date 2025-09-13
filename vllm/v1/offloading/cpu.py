# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from collections.abc import Iterator
from typing import Optional

import torch

from vllm.attention import get_attn_backend
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.offloading.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.offloading.lru_manager import (Backend, BlockStatus,
                                            LRUOffloadingManager)
from vllm.v1.offloading.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.offloading.spec import OffloadingSpec
from vllm.v1.offloading.worker.cpu_gpu import CpuGpuOffloadingHandler
from vllm.v1.offloading.worker.worker import OffloadingHandler


class CPUBlockStatus(BlockStatus):
    _fields_ = BlockStatus._fields_ + [("block_id", ctypes.c_int64)
                                       ]  # type: ignore

    def __init__(self, block_id: int):
        super().__init__()
        self.block_id = block_id

    def get_load_store_spec(self, block_hash: int) -> CPULoadStoreSpec:
        return CPULoadStoreSpec(self.block_id)


class CPUBackend(Backend):

    def __init__(self, block_size: int, num_blocks: int):
        super().__init__(block_size=block_size,
                         medium=CPULoadStoreSpec.medium())

        self.num_blocks: int = num_blocks
        self.num_allocated_blocks: int = 0
        self.allocated_blocks_free_list: list[int] = []

    def get_num_free_blocks(self):
        return (len(self.allocated_blocks_free_list) + self.num_blocks -
                self.num_allocated_blocks)

    def allocate_blocks(self, block_hashes: list[int]) -> list[BlockStatus]:
        num_fresh_blocks = min(len(block_hashes),
                               self.num_blocks - self.num_allocated_blocks)
        num_reused_blocks = len(block_hashes) - num_fresh_blocks
        assert len(self.allocated_blocks_free_list) >= num_reused_blocks

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh_blocks):
            blocks.append(CPUBlockStatus(self.num_allocated_blocks))
            self.num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused_blocks):
            block_id = self.allocated_blocks_free_list.pop()
            blocks.append(CPUBlockStatus(block_id))

        return blocks

    def free(self, block: BlockStatus):
        assert isinstance(block, CPUBlockStatus)
        self.allocated_blocks_free_list.append(block.block_id)


class CPUOffloadingSpec(OffloadingSpec):

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        self._num_cpu_blocks: Optional[int] = None

        # scheduler-side
        self._manager: Optional[OffloadingManager] = None

        # worker-side
        self._handler: Optional[OffloadingHandler] = None

    @property
    def num_cpu_blocks(self):
        if self._num_cpu_blocks is None:
            self._num_cpu_blocks = self.extra_config.get("num_cpu_blocks")
            if not self._num_cpu_blocks:
                raise Exception("num_cpu_blocks must be specified "
                                "in kv_connector_extra_config")
        return self._num_cpu_blocks

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (kv_events_config is not None
                             and kv_events_config.enable_kv_cache_events)
            self._manager = LRUOffloadingManager(CPUBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_cpu_blocks),
                                                 enable_events=enable_events)
        return self._manager

    def get_handlers(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec],
                        OffloadingHandler]]:
        if not self._handler:
            if not current_platform.is_cuda():
                raise Exception("CPU Offloading is currently only supported"
                                " on CUDA GPUs")
            attn_backend = get_attn_backend(
                self.vllm_config.model_config.get_head_size(),
                self.vllm_config.model_config.dtype,
                self.vllm_config.cache_config.cache_dtype,
                self.gpu_block_size,
                self.vllm_config.model_config.is_attention_free,
                use_mla=self.vllm_config.model_config.use_mla)

            self._handler = CpuGpuOffloadingHandler(
                attn_backend=attn_backend,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=self.num_cpu_blocks,
                gpu_caches=list(kv_caches.values()))

        assert self._handler is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handler
