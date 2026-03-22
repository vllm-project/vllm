# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载后端抽象模块。

本模块定义了 KV 卸载后端的抽象接口，负责：
- 分配和释放 KV 数据块的存储空间
- 提供块状态管理（引用计数、就绪状态）
- 生成后端特定的加载/存储规范

主要类：
- BlockStatus: 单个 KV 数据块的卸载状态
- Backend: 后端抽象基类
"""

import ctypes
from abc import ABC, abstractmethod
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec


class BlockStatus(ctypes.Structure):
    """单个 KV 数据块的卸载状态。

    持有以下信息：
    - ref_cnt: 当前使用此块作为源的传输数量
      值为 -1 表示块尚未准备好被读取
    - load_store_spec: 后端特定的信息，关于如何实际读取/写入块

    Attributes:
        ref_cnt: 引用计数，-1 表示未就绪，>=0 表示就绪
    """

    _fields_ = [("ref_cnt", ctypes.c_int32)]

    def __init__(self):
        """初始化块状态。

        将块初始化为"未就绪"状态（ref_cnt = -1）。
        """
        super().__init__()
        # 初始化块为"未就绪"（ref_cnt = -1）
        self.ref_cnt = -1

    @property
    def is_ready(self) -> bool:
        """返回块是否已准备好被读取。

        Returns:
            如果 ref_cnt >= 0 则返回 True，表示块已就绪
        """
        return self.ref_cnt >= 0


class Backend(ABC):
    """后端抽象基类。

    用于分配空间并为写入 KV 块到某个后端生成规范。
    每个具体的后端实现需要定义如何分配、释放和定位块。
    """

    def __init__(self, block_size: int, medium: str):
        """初始化后端。

        Args:
            block_size: 块大小（字节）
            medium: 存储介质类型字符串
        """
        self.block_size = block_size
        self.medium = medium

    @abstractmethod
    def get_num_free_blocks(self):
        """返回当前可分配的块数量。

        Returns:
            当前可用的空闲块数量
        """
        pass

    @abstractmethod
    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        """分配空间用于写入块。

        此方法假设有足够的空间进行分配。
        在未先检查 get_num_free_blocks 的情况下使用是不安全的。

        Args:
            block_hashes: 用于标识要写入的块的哈希列表

        Returns:
            分配块的 BlockStatus 列表。
            每个返回项的 ref_cnt 将为 -1，表示块尚未准备好被读取。
        """
        pass

    @abstractmethod
    def free(self, block: BlockStatus):
        """释放之前分配的块。

        只对 allocate_blocks 返回的块调用此函数，
        并且每个块只能调用一次。

        Args:
            block: 要释放的块
        """
        pass

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        """获取后端特定的块读写信息。

        Args:
            block_hashes: 标识块的哈希列表
            blocks: 块状态列表

        Returns:
            LoadStoreSpec 对象，worker 可使用它来读取/写入块。

        Raises:
            NotImplementedError: 如果后端未实现此方法
        """
        raise NotImplementedError
