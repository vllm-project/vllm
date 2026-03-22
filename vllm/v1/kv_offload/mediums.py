# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载介质定义模块。

本模块定义了 KV 卸载的介质类型和相应的加载/存储规范，负责：
- 定义基于块 ID 的加载/存储规范基类
- 提供 GPU 和 CPU 介质的加载/存储规范实现

主要类：
- BlockIDsLoadStoreSpec: 基于块 ID 的加载/存储规范基类
- GPULoadStoreSpec: GPU 内存加载/存储规范
- CPULoadStoreSpec: CPU 内存加载/存储规范
"""

from abc import ABC
from collections.abc import Sequence

import numpy as np

from vllm.v1.kv_offload.abstract import LoadStoreSpec


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
        """返回块 ID 数组的字符串表示。

        Returns:
            块 ID 数组的字符串表示
        """
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
