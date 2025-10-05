# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from typing import Optional

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandler
from vllm.v1.kv_offload.worker.worker import OffloadingHandler


class CPUOffloadingSpec(OffloadingSpec):

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        swap_space_bytes = self.extra_config.get("swap_space_bytes")
        if not swap_space_bytes:
            raise Exception("swap_space_bytes must be specified "
                            "in kv_connector_extra_config")
        self.swap_space_bytes: int = swap_space_bytes

        # scheduler-side
        self._manager: Optional[OffloadingManager] = None

        # worker-side
        self._handler: Optional[OffloadingHandler] = None

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_bytes_per_offloaded_block = (
                self.vllm_config.cache_config.kv_bytes_per_block *
                (self.offloaded_block_size // self.gpu_block_size))
            num_blocks = self.swap_space_bytes // kv_bytes_per_offloaded_block
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (kv_events_config is not None
                             and kv_events_config.enable_kv_cache_events)
            self._manager = LRUOffloadingManager(CPUBackend(
                block_size=self.offloaded_block_size, num_blocks=num_blocks),
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

            layer_names = list(kv_caches.keys())
            layers = get_layers_from_vllm_config(self.vllm_config,
                                                 AttentionLayerBase,
                                                 layer_names)
            attn_backends = {
                layer_name: layers[layer_name].get_attn_backend()
                for layer_name in layer_names
            }

            kv_bytes_per_offloaded_block = (
                self.vllm_config.cache_config.kv_bytes_per_block *
                (self.offloaded_block_size // self.gpu_block_size))
            num_blocks = self.swap_space_bytes // kv_bytes_per_offloaded_block

            self._handler = CpuGpuOffloadingHandler(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=num_blocks,
                gpu_caches=kv_caches)

        assert self._handler is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handler
