# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载规范模块。

本模块定义了 KV 卸载的规范类，负责：
- 定义加载/存储规范的基类
- 提供 GPU 和 CPU 加载/存储规范实现
- 定义卸载规范的抽象接口

主要类：
- BlockIDsLoadStoreSpec: 基于块 ID 的加载/存储规范基类
- GPULoadStoreSpec: GPU 内存加载/存储规范
- CPULoadStoreSpec: CPU 内存加载/存储规范
- OffloadingSpec: 卸载规范抽象基类
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """基于块 ID 的加载/存储规范基类。

    用于从给定的块号加载/存储 KV 块。
    使用 numpy 数组存储块 ID，便于高效处理。

    Attributes:
        block_ids: 块 ID 的 numpy 数组（int64 类型）
    """

    def __init__(self, block_ids: list[int]):
        """初始化块 ID 加载/存储规范。

        Args:
            block_ids: 块 ID 列表
        """
        self.block_ids = np.array(block_ids, dtype=np.int64)

    def __repr__(self) -> str:
        return repr(self.block_ids)


class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """GPU 内存加载/存储规范。

    用于从 GPU 内存加载/存储 KV 块。

    如果有多个 KV 组，块期望按组索引排序。
    在这种情况下，group_sizes[i] 决定第 i 个 KV 组的块数量，
    因此 sum(group_sizes) == len(block_ids)。
    group_sizes=None 表示单个 KV 组。

    如果给出 block_indices，每组块 ID（由 group_sizes 决定）将对应
    逻辑上连续的块，例如某个请求的块 5-10。
    block_indices[i] 将表示组#i 的第一个块的块索引。
    因此，len(block_indices) == len(group_sizes) = KV 缓存组数量。

    此信息对于支持从比 GPU 块更大的已卸载块加载是必需的。
    在这种情况下，每组第一个 GPU 块可能与卸载块大小未对齐，
    因此知道 block_indices[i] 允许 worker 正确跳过每个组第一个
    匹配卸载块的部分。

    从 GPU 卸载总是与卸载块大小对齐，因此 block_indices 仅在
    加载到 GPU 时由卸载连接器设置。

    Attributes:
        group_sizes: 每组的块大小序列
        block_indices: 每组的块索引序列（可选）
    """

    def __init__(
        self,
        block_ids: list[int],
        group_sizes: Sequence[int],
        block_indices: Sequence[int] | None = None,
    ):
        """初始化 GPU 加载/存储规范。

        Args:
            block_ids: 块 ID 列表
            group_sizes: 每组的块数量序列
            block_indices: 每组的块索引序列（可选）

        Raises:
            AssertionError: 如果参数不满足约束条件
        """
        super().__init__(block_ids)
        assert sum(group_sizes) == len(block_ids)
        assert block_indices is None or len(block_indices) == len(group_sizes)
        self.group_sizes: Sequence[int] = group_sizes
        self.block_indices: Sequence[int] | None = block_indices

    @staticmethod
    def medium() -> str:
        """返回介质类型。

        Returns:
            "GPU"
        """
        return "GPU"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """CPU 内存加载/存储规范。

    用于从 CPU 内存加载/存储 KV 块。
    """

    @staticmethod
    def medium() -> str:
        """返回介质类型。

        Returns:
            "CPU"
        """
        return "CPU"


class OffloadingSpec(ABC):
    """卸载规范抽象基类。

    为卸载连接器定义规范接口。
    提供配置解析和管理器/处理程序获取方法。

    Attributes:
        vllm_config: vLLM 配置
        kv_cache_config: KV 缓存配置
        extra_config: KV 连接器额外配置
        hash_block_size: 用于前缀缓存哈希的块大小
        gpu_block_size: 每组的 GPU 块大小元组
        block_size_factor: 卸载块大小与 GPU 块大小的比率
    """

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        """初始化卸载规范。

        Args:
            vllm_config: vLLM 配置
            kv_cache_config: KV 缓存配置

        Note:
            此 API 是实验性的，随着设计的迭代可能会发生变化。
        """
        logger.warning(
            "Initializing OffloadingSpec. This API is experimental and "
            "subject to change in the future as we iterate the design."
        )
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config

        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.extra_config = kv_transfer_config.kv_connector_extra_config

        # vLLM 用于哈希请求 token 的块大小，以实现前缀缓存
        self.hash_block_size = vllm_config.cache_config.block_size
        # 每组的 GPU 块大小
        self.gpu_block_size: tuple[int, ...] = tuple(
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        )

        for block_size in self.gpu_block_size:
            assert block_size % self.hash_block_size == 0

        # offloaded_block_size / gpu_block_size
        self.block_size_factor: int = 1

        offloaded_block_size = self.extra_config.get("block_size")
        if offloaded_block_size is not None:
            offloaded_block_size_int = int(offloaded_block_size)
            gpu_block_sizes = set(self.gpu_block_size)
            assert len(gpu_block_sizes) == 1, (
                "如果在 kv_connector_extra_config 中指定了 'block_size'，"
                "必须至少有一个 KV 缓存组，"
                "并且所有组必须有相同的块大小。"
            )
            gpu_block_size = gpu_block_sizes.pop()

            assert offloaded_block_size_int % gpu_block_size == 0
            self.block_size_factor = offloaded_block_size_int // gpu_block_size

    @abstractmethod
    def get_manager(self) -> OffloadingManager:
        """获取卸载管理器。

        获取 OffloadingManager，调度器端的卸载连接器将使用它
        来跟踪已卸载的块并管理驱逐。

        Returns:
            卸载管理器实例
        """
        pass

    @abstractmethod
    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        """获取卸载处理程序及其源/目标类型。

        Args:
            kv_caches: layer_name -> gpu_kv_cache 张量的字典
            attn_backends: layer_name -> AttentionBackend 的字典

        Yields:
            (src_type, dst_type, offloading_handler) 元组
        """
        pass
