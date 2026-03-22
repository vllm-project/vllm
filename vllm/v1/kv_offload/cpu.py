# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载 CPU 规范模块。

本模块实现了 CPU 卸载规范，负责：
- 解析 CPU 卸载配置
- 计算 CPU 块数量和大小
- 创建 LRU/ARC 驱逐策略的卸载管理器
- 提供 CPU-GPU 双向传输处理程序

主要类：
- CPUOffloadingSpec: CPU 内存卸载规范实现
"""

from collections.abc import Iterator

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.reuse_manager import FilterReusedOffloadingManager
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler


class CPUOffloadingSpec(OffloadingSpec):
    """CPU 内存卸载规范实现。

    支持多种驱逐策略（LRU/ARC）和重用频率过滤。
    配置参数通过 kv_connector_extra_config 传递。

    Attributes:
        num_blocks: CPU 缓存可容纳的块数量
        eviction_policy: 驱逐策略（"lru" 或 "arc"）
        _manager: 调度器侧的卸载管理器实例
        _handlers: worker 侧的卸载处理程序实例
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        """初始化 CPU 卸载规范。

        Args:
            vllm_config: vLLM 配置
            kv_cache_config: KV 缓存配置

        Raises:
            Exception: 如果未指定 cpu_bytes_to_use 参数
        """
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # 计算每个卸载块的 KV 字节数
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
        """获取卸载管理器。

        根据配置的驱逐策略创建相应的管理器实例。
        如果配置了重用频率过滤，则包装一层 FilterReusedOffloadingManager。

        Returns:
            卸载管理器实例
        """
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]
            offloaded_block_size = gpu_block_size * self.block_size_factor
            backend = CPUBackend(
                block_size=offloaded_block_size, num_blocks=self.num_blocks
            )

            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. "
                    f"Supported policies: lru, arc"
                )

            # store_threshold: 块必须在 lookup() 中出现的次数
            # 才有资格进行 CPU 卸载。值 < 2 禁用过滤
            # （阈值 1 等于无过滤；0 是默认值）
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            if store_threshold >= 2:
                max_tracker_size = int(
                    self.extra_config.get("max_tracker_size", 64_000)
                )
                self._manager = FilterReusedOffloadingManager(
                    backing=self._manager,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        """获取卸载处理程序及其源/目标类型。

        创建 CPU-GPU 双向传输处理程序。

        Args:
            kv_caches: layer_name -> gpu_kv_cache 张量的字典
            attn_backends: layer_name -> AttentionBackend 的字典

        Yields:
            (src_type, dst_type, offloading_handler) 元组

        Raises:
            Exception: 如果当前平台不支持 CUDA
        """
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike GPUs"
                )

            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]

            self._handlers = CpuGpuOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=gpu_block_size,
                cpu_block_size=gpu_block_size * self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
                gpu_caches=kv_caches,
            )

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler
