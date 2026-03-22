# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载重用过滤管理器模块。

本模块实现了重用频率过滤的卸载管理器装饰器，负责：
- 跟踪块的访问频率
- 过滤掉访问次数不足阈值的块，避免将其卸载
- 维护 LRU 计数器，限制跟踪器大小

主要类：
- FilterReusedOffloadingManager: 重用频率过滤的卸载管理器装饰器
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


class FilterReusedOffloadingManager(OffloadingManager):
    """OffloadingManager 装饰器，过滤掉重用频率低于阈值的块。

    此装饰器拦截两个关键方法：
    - ``lookup`` — 记录每个访问的块哈希到内部 LRU 计数器
    - ``prepare_store`` — 过滤掉未达到阈值的块，再调用底层管理器

    所有其他方法都委托给底层的 *backing* 管理器。

    访问计数机制：
        1. 每次调用 lookup() 时，块的计数增加
        2. 如果计数 >= store_threshold，块才有资格被卸载
        3. 计数器使用 OrderedDict 维护，超过 max_tracker_size 时驱逐 LRU 条目

    Args:
        backing: 底层的 ``OffloadingManager`` 实例
        store_threshold: 块必须在 ``lookup()`` 中至少出现这么多次
            才有资格进行卸载。必须 >= 2（值为 1 等同于无过滤）
        max_tracker_size: 内部 LRU 计数器的最大条目数

    Attributes:
        _backing: 底层卸载管理器
        store_threshold: 存储阈值
        max_tracker_size: 跟踪器最大大小
        counts: 块哈希到访问计数的有序字典
    """

    def __init__(
        self,
        backing: OffloadingManager,
        store_threshold: int = 2,
        max_tracker_size: int = 64_000,
    ):
        """初始化重用频率过滤管理器。

        Args:
            backing: 底层卸载管理器
            store_threshold: 存储阈值，必须 >= 2
            max_tracker_size: 跟踪器最大大小，必须 >= 1

        Raises:
            ValueError: 如果 store_threshold < 2 或 max_tracker_size < 1
        """
        if store_threshold < 2:
            raise ValueError(
                "FilterReusedOffloadingManager store_threshold must be >= 2, "
                f"got {store_threshold}"
            )
        if max_tracker_size < 1:
            raise ValueError(
                "FilterReusedOffloadingManager max_tracker_size must be >= 1, "
                f"got {max_tracker_size}"
            )
        self._backing = backing
        self.store_threshold = store_threshold
        self.max_tracker_size = max_tracker_size
        # Ordered 以便我们可以 O(1) 驱逐 LRU 条目
        self.counts: OrderedDict[BlockHash, int] = OrderedDict()

    # ------------------------------------------------------------------
    # 拦截的方法
    # ------------------------------------------------------------------

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """记录每个块的访问次数，然后委托给底层管理器。

        对于每个块哈希：
        - 如果已存在：移动到末尾并增加计数
        - 如果不存在：添加到末尾，计数设为 1
        - 如果超过 max_tracker_size：驱逐 LRU 条目（头部）

        Args:
            block_hashes: 用于标识块的哈希列表

        Returns:
            底层管理器的 lookup() 结果
        """
        block_hashes = list(block_hashes)
        for block_hash in block_hashes:
            if block_hash in self.counts:
                self.counts.move_to_end(block_hash)
                self.counts[block_hash] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)  # 驱逐 LRU
                    self.counts[block_hash] = 1

        return self._backing.lookup(block_hashes)

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """过滤掉低于阈值的块，然后委托给底层管理器。

        过滤在调用底层管理器的 ``prepare_store`` 之前进行，
        因此被过滤掉的块不会消耗 CPU 卸载容量。

        Args:
            block_hashes: 用于标识块的哈希列表

        Returns:
            底层管理器对过滤后块列表的 PrepareStoreOutput
        """
        block_hashes = list(block_hashes)
        eligible = [
            bh for bh in block_hashes if self.counts.get(bh, 0) >= self.store_threshold
        ]

        # 委托给底层管理器，仅传递符合条件的哈希
        # 传递空列表是安全且有意的 — LRUOffloadingManager 和
        # ARCOffloadingManager 都能正确处理，返回空列表的 PrepareStoreOutput
        return self._backing.prepare_store(eligible)

    # ------------------------------------------------------------------
    # 委托的方法
    # ------------------------------------------------------------------

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """委托给底层管理器。

        Args:
            block_hashes: 用于标识块的哈希列表

        Returns:
            底层管理器的 prepare_load() 结果
        """
        return self._backing.prepare_load(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        """委托给底层管理器。

        Args:
            block_hashes: 用于标识块的哈希列表
        """
        return self._backing.touch(block_hashes)

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        """委托给底层管理器。

        Args:
            block_hashes: 用于标识块的哈希列表
        """
        return self._backing.complete_load(block_hashes)

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        """委托给底层管理器。

        Args:
            block_hashes: 用于标识块的哈希列表
            success: 块是否成功存储
        """
        return self._backing.complete_store(block_hashes, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        """委托给底层管理器。

        Returns:
            底层管理器的 take_events() 结果
        """
        return self._backing.take_events()
