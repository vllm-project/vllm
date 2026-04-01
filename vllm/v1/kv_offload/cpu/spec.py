# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.attention_manager import (
    AttentionWeightedOffloadingManager,
)
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.hybrid_manager import HybridOffloadingManager
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.reuse_manager import FilterReusedOffloadingManager
from vllm.v1.kv_offload.spec import CanonicalKVCaches, OffloadingSpec
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler


class CPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # calculate kv_bytes_per_offloaded_block
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

        kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor
        self.num_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handlers: CpuGpuOffloadingHandlers | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            backend = CPUBackend(
                block_size=self.offloaded_block_size, num_blocks=self.num_blocks
            )

            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            elif self.eviction_policy == "attention":
                score_decay = float(
                    self.extra_config.get("score_decay", 0.95)
                )
                self._manager = AttentionWeightedOffloadingManager(
                    backend=backend,
                    enable_events=enable_events,
                    score_decay=score_decay,
                )
            elif self.eviction_policy == "hybrid":
                alpha = float(
                    self.extra_config.get("attention_weight", 0.5)
                )
                beta = float(
                    self.extra_config.get("recency_weight", 0.3)
                )
                gamma = float(
                    self.extra_config.get("frequency_weight", 0.2)
                )
                score_decay = float(
                    self.extra_config.get("score_decay", 0.95)
                )
                self._manager = HybridOffloadingManager(
                    backend=backend,
                    enable_events=enable_events,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    score_decay=score_decay,
                )
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. "
                    f"Supported policies: lru, arc, attention, hybrid"
                )
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike GPUs"
                )

            self._handlers = CpuGpuOffloadingHandlers(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
            )

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler
