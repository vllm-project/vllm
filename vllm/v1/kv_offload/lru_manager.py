# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载 LRU 管理器模块。

本模块实现了基于 LRU（最近最少使用）驱逐策略的卸载管理器，负责：
- 管理已卸载块的 LRU 顺序
- 提供块的查找、加载、存储功能
- 支持块驱逐和事件跟踪

主要类：
- LRUOffloadingManager: 基于 LRU 策略的卸载管理器
"""

from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus


class LRUOffloadingManager(OffloadingManager):
    """基于 LRU 驱逐策略的卸载管理器。

    使用 OrderedDict 维护块的 LRU 顺序，最近访问的块移动到末尾。
    驱逐时从头部（最久未使用）开始驱逐。

    Attributes:
        backend: 后端存储实例
        blocks: 块哈希到块状态的有序字典（LRU 队列）
        events: 卸载事件列表（如果启用事件跟踪）
    """

    def __init__(self, backend: Backend, enable_events: bool = False):
        """初始化 LRU 卸载管理器。

        Args:
            backend: 后端存储实例
            enable_events: 是否启用事件跟踪
        """
        self.backend: Backend = backend
        # block_hash -> BlockStatus，OrderedDict 维护 LRU 顺序
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """查找从第一个块开始的最大连续卸载块序列长度。

        按顺序遍历块哈希，直到遇到不在缓存中或未就绪的块。

        Args:
            block_hashes: 用于标识块的哈希列表

        Returns:
            连续命中的块数量
        """
        hit_count = 0
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """准备加载指定块。

        增加块的引用计数，防止在加载过程中被驱逐。

        Args:
            block_hashes: 用于标识块的哈希列表

        Returns:
            LoadStoreSpec 对象，worker 可使用它来定位和加载块
        """
        blocks = []
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.is_ready
            block.ref_cnt += 1
            blocks.append(block)

        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        """标记块为最近使用。

        将块移动到 LRU 队列末尾（最近使用位置）。
        按反向顺序处理，保持原始相对顺序。

        Args:
            block_hashes: 用于标识块的哈希列表
        """
        for block_hash in reversed(list(block_hashes)):
            if self.blocks.get(block_hash):
                self.blocks.move_to_end(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        """标记之前准备加载的块为已完成加载。

        减少块的引用计数，允许块再次被驱逐。

        Args:
            block_hashes: 用于标识块的哈希列表
        """
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """准备将指定块卸载。

        1. 过滤掉已经存储的块
        2. 计算需要驱逐的块数量
        3. 从 LRU 队列头部驱逐最久未使用的块
        4. 分配新块并返回存储规范

        Args:
            block_hashes: 用于标识块的哈希列表

        Returns:
            PrepareStoreOutput 对象，包含要存储的块、存储位置和驱逐的块列表
            如果无法驱逐足够的块则返回 None
        """
        block_hashes_list = list(block_hashes)

        # 过滤掉已经存储的块
        block_hashes_to_store = [
            block_hash
            for block_hash in block_hashes_list
            if block_hash not in self.blocks
        ]

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        # 构建要驱逐的块列表
        to_evict = []
        if num_blocks_to_evict > 0:
            # 来自原始输入的块被排除在驱逐候选之外：
            # 已经存储的块必须在此调用后保留在缓存中
            protected = set(block_hashes_list)
            for block_hash, block in self.blocks.items():
                if block.ref_cnt == 0 and block_hash not in protected:
                    to_evict.append(block_hash)
                    num_blocks_to_evict -= 1
                    if num_blocks_to_evict == 0:
                        break
            else:
                # 无法驱逐足够的块
                return None

        # 驱逐块
        for block_hash in to_evict:
            self.backend.free(self.blocks.pop(block_hash))

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.blocks[block_hash] = block

        # 为分配的块构建存储规范
        store_spec = self.backend.get_load_store_spec(block_hashes_to_store, blocks)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        """标记之前准备存储的块为已存储。

        如果成功，将块的引用计数设置为 0，使其变为可加载状态。
        如果失败，释放块并从缓存中移除。

        Args:
            block_hashes: 用于标识块的哈希列表
            success: 块是否成功存储
        """
        stored_block_hashes: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    self.backend.free(block)
                    del self.blocks[block_hash]

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        """从管理器获取卸载事件。

        Yields:
            自上次调用以来收集的新 OffloadingEvent 对象
        """
        if self.events is not None:
            yield from self.events
            self.events.clear()
