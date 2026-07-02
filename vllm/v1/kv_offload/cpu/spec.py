# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from typing_extensions import override

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = 1

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            )
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = vllm_config.parallel_config.world_size
        self.num_blocks = 0
        self.kv_bytes_per_offloaded_block = 0
        self.cpu_page_size_per_worker = 0
        assert kv_cache_config is not None
        if kv_cache_config.num_blocks > 0 and world_size > 0:
            is_packed = any(t.block_stride for t in kv_cache_config.kv_cache_tensors)
            assert not is_packed or all(
                t.block_stride for t in kv_cache_config.kv_cache_tensors
            )
            total_gpu_kv_bytes = (
                kv_cache_config.kv_cache_tensors[0].size
                if is_packed
                else sum(t.size for t in kv_cache_config.kv_cache_tensors)
            )
            total_gpu_kv_bytes = max(
                total_gpu_kv_bytes,
                self._get_per_layer_gpu_kv_bytes(kv_cache_config),
            )
            kv_bytes_per_block = (
                total_gpu_kv_bytes // kv_cache_config.num_blocks
            ) * world_size
            kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_offloaded_block // world_size

            # calculate num_blocks
            aligned_kv_bytes_per_offloaded_block = round_up(
                kv_bytes_per_offloaded_block, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = (
                int(cpu_bytes_to_use) // aligned_kv_bytes_per_offloaded_block
            )

            # Expose aligned_kv_bytes_per_offloaded_block as
            # kv_bytes_per_offloaded_block. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            self.kv_bytes_per_offloaded_block = aligned_kv_bytes_per_offloaded_block

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    @staticmethod
    def _get_per_layer_gpu_kv_bytes(kv_cache_config: KVCacheConfig) -> int:
        """Return a conservative per-layer KV byte upper bound.

        KVCacheTensor entries can share storage across layers, but canonical
        offloading tensors are split whenever their concrete tensor views differ
        in stride or offset. This upper bound keeps CPU offload rows large
        enough for those split views.
        """
        total_page_bytes = 0
        for group in kv_cache_config.kv_cache_groups:
            group_spec = group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                per_layer_specs = group_spec.kv_cache_specs
            else:
                per_layer_specs = {}
            for layer_name in group.layer_names:
                layer_spec = per_layer_specs.get(layer_name, group_spec)
                total_page_bytes += layer_spec.page_size_bytes
        return total_page_bytes * kv_cache_config.num_blocks

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
        )

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
