# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载后端模块。

本模块提供了 KV 卸载后端的实现，负责：
- 管理后端特定的块状态
- 分配和释放块空间
- 生成加载/存储规范

主要类：
- CPUBlockStatus: CPU 后端块状态
- CPUBackend: CPU 内存后端实现
"""

import ctypes
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec


class CPUBlockStatus(BlockStatus):
    """CPU 后端块状态。

    继承自 BlockStatus，增加 block_id 字段用于标识 CPU 块。

    Attributes:
        ref_cnt: 引用计数（继承自 BlockStatus）
        block_id: CPU 块 ID
    """

    _fields_ = BlockStatus._fields_ + [("block_id", ctypes.c_int64)]  # type: ignore

    def __init__(self, block_id: int):
        """初始化 CPU 块状态。

        Args:
            block_id: CPU 块 ID
        """
        super().__init__()
        self.block_id = block_id


class CPUBackend(Backend):
    """CPU 内存后端实现。

    管理 CPU 块的分配和释放，维护空闲块列表以实现高效复用。

    Attributes:
        num_blocks: CPU 块总数
        num_allocated_blocks: 已分配的块数量
        allocated_blocks_free_list: 已分配块的自由列表（可复用）
    """

    def __init__(self, block_size: int, num_blocks: int):
        """初始化 CPU 后端。

        Args:
            block_size: 块大小（字节）
            num_blocks: CPU 块总数
        """
        super().__init__(block_size=block_size, medium=CPULoadStoreSpec.medium())

        self.num_blocks: int = num_blocks
        self.num_allocated_blocks: int = 0
        self.allocated_blocks_free_list: list[int] = []

    def get_num_free_blocks(self):
        """返回当前可分配的块数量。

        计算方式：空闲列表中的块数 + 未分配的块数

        Returns:
            当前可用的空闲块数量
        """
        return (
            len(self.allocated_blocks_free_list)
            + self.num_blocks
            - self.num_allocated_blocks
        )

    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        """分配空间用于写入块。

        分配策略：
        1. 优先从已释放的块中复用（allocated_blocks_free_list）
        2. 如果不够，分配新的块（从 num_allocated_blocks 递增）

        Args:
            block_hashes: 用于标识要写入的块的哈希列表

        Returns:
            分配块的 CPUBlockStatus 列表

        Raises:
            AssertionError: 如果空闲列表中的块不足以复用
        """
        num_fresh_blocks = min(
            len(block_hashes), self.num_blocks - self.num_allocated_blocks
        )
        num_reused_blocks = len(block_hashes) - num_fresh_blocks
        assert len(self.allocated_blocks_free_list) >= num_reused_blocks

        # 分配新块
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh_blocks):
            blocks.append(CPUBlockStatus(self.num_allocated_blocks))
            self.num_allocated_blocks += 1

        # 分配复用的块
        for _ in range(num_reused_blocks):
            block_id = self.allocated_blocks_free_list.pop()
            blocks.append(CPUBlockStatus(block_id))

        return blocks

    def free(self, block: BlockStatus):
        """释放之前分配的块。

        将块 ID 添加到空闲列表，以便后续复用。

        Args:
            block: 要释放的块（必须是 CPUBlockStatus 类型）

        Raises:
            AssertionError: 如果块不是 CPUBlockStatus 类型
        """
        assert isinstance(block, CPUBlockStatus)
        self.allocated_blocks_free_list.append(block.block_id)

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        """获取 CPU 后端特定的块读写信息。

        从块状态列表中提取块 ID，创建 CPULoadStoreSpec。

        Args:
            block_hashes: 标识块的哈希列表（未使用）
            blocks: 块状态列表

        Returns:
            CPULoadStoreSpec 对象，包含块 ID 列表
        """
        return CPULoadStoreSpec([block.block_id for block in blocks])
