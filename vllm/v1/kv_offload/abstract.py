# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 数据卸载管理器模块。

本模块定义了 vLLM V1 中 KV 数据卸载的管理类，负责：
- 跟踪哪些块已卸载及其地址
- 提供查找、加载、存储块的接口
- 管理块的 LRU（最近最少使用）策略
- 保护块在传输过程中不被驱逐

主要类：
- LoadStoreSpec: 加载/存储规范的抽象基类
- PrepareStoreOutput: 存储准备输出数据类
- OffloadingEvent: 卸载事件数据类
- OffloadingManager: 卸载管理器抽象基类

使用说明：
    OffloadingManager 类在调度器中运行，提供以下原语：
    - lookup() - 查找从第一个块开始的最大连续卸载块序列长度
    - prepare_load() - 准备加载指定块，保护它们不被驱逐
    - touch() - 标记块为最近使用，用于 LRU 跟踪
    - complete_load() - 完成加载，允许块再次被驱逐
    - prepare_store() - 准备存储指定块，返回被驱逐的块列表
    - complete_store() - 完成存储，使块变为可加载状态
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash


class LoadStoreSpec(ABC):
    """加载/存储规范抽象基类。

    封装允许 worker 加载和可选存储 KV 数据块的元数据。
    每个具体的实现类需要提供 medium() 方法来标识目标存储介质类型。
    """

    @staticmethod
    @abstractmethod
    def medium() -> str:
        """返回此存储/加载目标的介质类型字符串表示。

        Returns:
            介质类型字符串（如 "cpu", "disk" 等）
        """
        pass


@dataclass
class PrepareStoreOutput:
    """存储准备输出数据类。

    封装准备存储操作的结果信息。

    Attributes:
        block_hashes_to_store: 需要存储的块哈希列表
        store_spec: 加载/存储规范
        block_hashes_evicted: 作为结果被驱逐的块哈希列表
    """

    block_hashes_to_store: list[BlockHash]
    store_spec: LoadStoreSpec
    block_hashes_evicted: list[BlockHash]


@dataclass
class OffloadingEvent:
    """卸载事件数据类。

    记录 KV 数据卸载事件的详细信息。

    Attributes:
        block_hashes: 涉及的块哈希列表
        block_size: 块大小
        medium: 存储介质类型
        removed: True 表示块被移除，False 表示块被存储
    """

    block_hashes: list[BlockHash]
    block_size: int
    medium: str
    # True 表示块被移除，False 表示块被存储
    removed: bool


class OffloadingManager(ABC):
    """卸载管理器抽象基类。

    在调度器中运行，跟踪哪些块已卸载及其地址。
    提供加载和存储 KV 数据块的完整生命周期管理。
    """

    @abstractmethod
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """查找最大连续卸载块序列长度。

        从第一个块开始，查找所有块都已卸载的最大连续序列长度。

        Args:
            block_hashes: 用于标识要查找的块的哈希列表

        Returns:
            当前已卸载的块的最大数量，如果查找应该稍后重试则返回 None。
            返回 None 将延迟 vLLM 调度器对该请求的处理。
        """
        pass

    @abstractmethod
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """准备加载指定块。

        准备的块在调用 complete_load 之前将受到保护，不会被驱逐。
        此方法假设所有给定的块都已卸载。

        Args:
            block_hashes: 用于标识块的哈希列表

        Returns:
            LoadStoreSpec 对象，worker 可使用它来定位和加载
            实际的已卸载 KV 数据。
        """
        pass

    def touch(self, block_hashes: Iterable[BlockHash]):
        """标记块为最近使用。

        实际上可能意味着将它们移动到 LRU 列表的末尾。
        此方法与 prepare_load 分离，以便即使是不需要从缓存读取的块
        （如 GPU 前缀缓存中的块）也能设置其新鲜度。

        Args:
            block_hashes: 用于标识块的哈希列表
        """
        return

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        """标记之前准备加载的块为已完成加载。

        调用此方法后，块将再次允许被驱逐。

        Args:
            block_hashes: 用于标识块的哈希列表
        """
        return

    @abstractmethod
    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """准备将指定块卸载。

        准备的块在调用 complete_store 之前将受到保护，不会被驱逐。

        Args:
            block_hashes: 用于标识块的哈希列表

        Returns:
            PrepareStoreOutput 对象，指示哪些块需要存储、
            存储位置（LoadStoreSpec），以及作为结果被驱逐的块列表。
            如果无法存储块，则返回 None。
        """
        pass

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        """标记之前准备存储的块为已存储。

        调用此方法后，块将变为可加载状态。
        如果 success 为 False，未标记为已存储的块将被移除。

        Args:
            block_hashes: 用于标识块的哈希列表
            success: 块是否成功存储
        """
        return

    def take_events(self) -> Iterable[OffloadingEvent]:
        """从管理器获取卸载事件。

        Yields:
            自上次调用以来收集的新 OffloadingEvent 对象。
        """
        return ()
