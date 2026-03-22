# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载 ARC 管理器模块。

本模块实现了基于 ARC（Adaptive Replacement Cache，自适应替换缓存）策略的卸载管理器，
负责：
- 维护 T1（最近访问）、T2（频繁访问）缓存队列
- 维护 B1、B2 幽灵列表（记录被驱逐的块）
- 自适应调整 T1 和 T2 的大小比例
- 根据访问模式动态优化驱逐策略

主要类：
- ARCOffloadingManager: 基于 ARC 策略的卸载管理器
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


class ARCOffloadingManager(OffloadingManager):
    """基于 ARC（自适应替换缓存）驱逐策略的卸载管理器。

    数据结构：
        T1: 最近缓存，包含只访问过一次的块
        T2: 频繁缓存，包含访问过多次的块
        B1/B2: 幽灵列表，记录从 T1/T2 驱逐的块（仅跟踪哈希）
        target_t1_size: T1 分区的自适应目标大小

    算法流程：
        1. 缓存查找（lookup）：
           搜索 T1 和 T2 中的块哈希，统计连续命中次数，
           直到遇到未命中或未就绪的块。

        2. 缓存触摸（touch）- 自适应学习：
           对每个块哈希（按反向顺序）：
           - 如果在 T1：移动到 T2（从最近访问提升到频繁访问）
           - 如果在 T2：移动到 MRU 位置（队列末尾）
           - 如果在 B1 幽灵列表：增加 target_t1_size
           - 如果在 B2 幽灵列表：减少 target_t1_size

        3. 块驱逐（prepare_store）- 自适应替换：
           根据自适应目标决定驱逐源：
           - 如果 T1 大小 > target_t1_size：从 T1 驱逐，加入 B1
           - 否则：从 T2 驱逐，加入 B2
           最后，限制每个幽灵列表的大小。

        4. 块插入（prepare_store）：
           新块总是插入到 T1，如果存在于 B1/B2 则移除。
           块可能在后续的 touch 操作中被提升到 T2。

    自适应行为：
        算法自我调整近期性与频率的权衡：
        - B1 命中：近期访问模式更重要 → 增加 T1
        - B2 命中：频繁访问模式更重要 → 减少 T1

    Attributes:
        backend: 后端存储实例
        target_t1_size: T1 缓存的目标大小（自适应调整）
        t1: 最近缓存队列（块哈希到块状态）
        t2: 频繁缓存队列（块哈希到块状态）
        b1: T1 的幽灵列表（仅记录块哈希）
        b2: T2 的幽灵列表（仅记录块哈希）
        events: 卸载事件列表（如果启用事件跟踪）
        cache_capacity: 缓存容量
    """

    def __init__(self, backend: Backend, enable_events: bool = False):
        """初始化 ARC 卸载管理器。

        Args:
            backend: 后端存储实例
            enable_events: 是否启用事件跟踪
        """
        self.backend: Backend = backend
        self.target_t1_size: float = 0.0
        self.t1: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.t2: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        # block_hash -> None（只关心是否存在）
        self.b1: OrderedDict[BlockHash, None] = OrderedDict()
        self.b2: OrderedDict[BlockHash, None] = OrderedDict()
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        self.cache_capacity: int = self.backend.get_num_free_blocks()

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """查找从第一个块开始的最大连续卸载块序列长度。

        在 T1 和 T2 中搜索块哈希，统计连续命中次数。

        Args:
            block_hashes: 用于标识块的哈希列表

        Returns:
            连续命中的块数量
        """
        hit_count = 0
        for block_hash in block_hashes:
            block = self.t1.get(block_hash) or self.t2.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """准备加载指定块。

        在 T1 或 T2 中查找块，增加引用计数。

        Args:
            block_hashes: 用于标识块的哈希列表

        Returns:
            LoadStoreSpec 对象，worker 可使用它来定位和加载块

        Raises:
            AssertionError: 如果块不在缓存中或未就绪
        """
        blocks = []
        for block_hash in block_hashes:
            block = self.t1.get(block_hash) or self.t2.get(block_hash)
            assert block is not None, f"Block {block_hash!r} not found in cache"
            assert block.is_ready, f"Block {block_hash!r} is not ready for reading"

            block.ref_cnt += 1
            blocks.append(block)

        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        """标记块为最近使用，实现自适应学习。

        按反向顺序处理每个块哈希：
        - T1 中的块：移动到 T2（从最近访问提升到频繁访问）
        - T2 中的块：移动到 MRU 位置（队列末尾）
        - B1 中的块：增加 T1 目标大小（近期访问模式更重要）
        - B2 中的块：减少 T1 目标大小（频繁访问模式更重要）

        Args:
            block_hashes: 用于标识块的哈希列表
        """
        for block_hash in reversed(list(block_hashes)):
            if block_hash in self.t1:
                block = self.t1.pop(block_hash)
                if not block.is_ready:
                    # 块刚刚准备好存储，并非真正被访问两次
                    # 保留在 T1 并标记为最近使用
                    self.t1[block_hash] = block
                else:
                    self.t2[block_hash] = block

            elif block_hash in self.t2:
                self.t2.move_to_end(block_hash)

            elif block_hash in self.b1:
                delta = max(1, len(self.b2) / len(self.b1))
                self.target_t1_size = min(
                    self.target_t1_size + delta, self.cache_capacity
                )
                # 移动到 MRU 位置（末尾）以保持其在幽灵列表中的新鲜度
                self.b1.move_to_end(block_hash)

            elif block_hash in self.b2:
                delta = max(1, len(self.b1) / len(self.b2))
                self.target_t1_size = max(self.target_t1_size - delta, 0)
                # 移动到 MRU 位置（末尾）以保持其在幽灵列表中的新鲜度
                self.b2.move_to_end(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        """标记之前准备加载的块为已完成加载。

        减少块的引用计数，允许块再次被驱逐。

        Args:
            block_hashes: 用于标识块的哈希列表

        Raises:
            AssertionError: 如果块不在缓存中或引用计数已为 0
        """
        for block_hash in block_hashes:
            block = self.t1.get(block_hash) or self.t2.get(block_hash)
            assert block is not None, f"Block {block_hash!r} not found"
            assert block.ref_cnt > 0, f"Block {block_hash!r} ref_cnt is already 0"

            block.ref_cnt -= 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """准备将指定块卸载。

        1. 过滤掉已经在 T1 或 T2 中的块
        2. 计算需要驱逐的块数量
        3. 根据自适应目标决定从 T1 还是 T2 驱逐
        4. 驱逐被保护块（原始输入中的块）以外的块
        5. 限制幽灵列表大小
        6. 分配新块并插入 T1

        Args:
            block_hashes: 用于标识块的哈希列表

        Returns:
            PrepareStoreOutput 对象，包含要存储的块、存储位置和驱逐的块列表
            如果无法驱逐足够的块则返回 None
        """
        block_hashes_list = list(block_hashes)

        block_hashes_to_store = []
        for block_hash in block_hashes_list:
            if block_hash not in self.t1 and block_hash not in self.t2:
                block_hashes_to_store.append(block_hash)

        if not block_hashes_to_store:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=self.backend.get_load_store_spec([], []),
                block_hashes_evicted=[],
            )

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        to_evict = []
        if num_blocks_to_evict > 0:
            # 来自原始输入的块被排除在驱逐候选之外：
            # 已经存储的块必须在此调用后保留在缓存中
            protected = set(block_hashes_list)
        while num_blocks_to_evict > 0:
            block_to_evict = None
            if len(self.t1) >= int(self.target_t1_size):
                # 尝试驱逐 T1 中最久未使用的块（头部）
                for block_hash, block in self.t1.items():
                    if block.ref_cnt == 0 and block_hash not in protected:
                        block_to_evict = (block_hash, block)
                        eviction_t = self.t1
                        eviction_b = self.b1
                        break
            if not block_to_evict:
                # 尝试驱逐 T2 中最久未使用的块（头部）
                for block_hash, block in self.t2.items():
                    if block.ref_cnt == 0 and block_hash not in protected:
                        block_to_evict = (block_hash, block)
                        eviction_t = self.t2
                        eviction_b = self.b2
                        break
                else:
                    # 无法驱逐足够的块，缓存已满
                    return None

            block_hash, block = block_to_evict
            del eviction_t[block_hash]
            eviction_b[block_hash] = None
            to_evict.append(block_hash)
            self.backend.free(block)
            num_blocks_to_evict -= 1

        # 限制幽灵列表大小不超过缓存容量
        for b in [self.b1, self.b2]:
            for i in range(len(b) - self.cache_capacity):
                b.popitem(last=False)

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
        assert len(blocks) == len(block_hashes_to_store), (
            "Backend did not allocate the expected number of blocks"
        )

        # 新块插入到 T1
        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.t1[block_hash] = block

            # 如果块在幽灵列表中，移除它
            self.b1.pop(block_hash, None)
            self.b2.pop(block_hash, None)

        store_spec = self.backend.get_load_store_spec(block_hashes_to_store, blocks)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        """标记之前准备存储的块为已存储。

        如果成功，将块的引用计数设置为 0，使其变为可加载状态。
        如果失败，从 T1 或 T2 中移除块并释放。

        Args:
            block_hashes: 用于标识块的哈希列表
            success: 块是否成功存储
        """
        stored_block_hashes: list[BlockHash] = []

        if success:
            for block_hash in block_hashes:
                block = self.t1.get(block_hash) or self.t2.get(block_hash)

                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.t1.pop(block_hash, None)

                if block is None:
                    block = self.t2.pop(block_hash, None)

                if block is not None and not block.is_ready:
                    self.backend.free(block)

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
