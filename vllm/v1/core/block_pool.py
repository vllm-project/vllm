# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 缓存块池模块。

本模块实现了 KV 缓存块池管理功能，负责：
- 管理 KV 缓存块的分配和释放
- 实现前缀缓存功能（通过块哈希映射）
- 处理块驱逐事件和 KV 缓存事件
- 支持块指标采集（用于性能分析）

主要类：
- BlockHashToBlockMap: 块哈希到块的映射表
- BlockPool: KV 缓存块池管理器
"""
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """块哈希到块的映射表。

    用于前缀缓存的块缓存数据结构，将块哈希映射到一个或多个 KVCacheBlock。
    大多数情况下一个块哈希对应单个 KVCacheBlock，但在存在重复块时可能对应多个。

    缓存的块是具有块哈希的完整块，可用于前缀缓存查找。
    这些块可能被运行中的请求使用，或者在 free_block_queue 中等待被驱逐。

    注意：
        1. 当前实现不去重缓存的块，即如果一个块已满并被缓存，不会检查
           缓存中是否已存在相同的块。这是为了确保分配的块 ID 不会改变，
           从而保证块表是只追加的。
        2. 使用联合类型（KVCacheBlock | dict）是为了减少内部字典带来的 GC 开销。

    Attributes:
        _cache: 内部缓存字典，键为 BlockHashWithGroupId，
                值为单个 KVCacheBlock 或块 ID 到块的字典
    """

    def __init__(self):
        """初始化块哈希到块的映射表。"""
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """获取具有给定块哈希键的任意一个块。

        Args:
            key: 块哈希键（包含组 ID）

        Returns:
            如果找到则返回 KVCacheBlock，否则返回 None
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """插入 KVCacheBlock 到缓存中。

        如果键已存在，则将新块合并到现有条目中（转为字典格式）。

        Args:
            key: 块哈希键（包含组 ID）
            block: 要插入的 KVCacheBlock
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # 键不存在时，将单个块附加到键
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # 如果已存在相同键的块，将原块和新块合并为字典
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # 如果已经是字典，直接插入新块
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """从缓存中弹出指定块 ID 的块。

        Args:
            key: 块哈希键（包含组 ID）
            block_id: 要弹出的块 ID

        Returns:
            如果找到则返回 KVCacheBlock，否则返回 None
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # 块哈希不在缓存中
            return None
        # TODO(Jialin): 如果找到键，block_id 应该总是存在于 blocks 中。
        # 目前为了安全起见保持原有行为。
        #
        # 后续可以添加 block_id == blocks.block_id 断言，
        # 并使用 del blocks[block_id] 替代。
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # 如果单个块的 ID 不匹配，将块放回缓存（这种情况应该很少见）
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # 尝试从块字典中弹出 block_id，如果字典仍包含块则放回缓存
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        """返回缓存中的条目数量。

        Returns:
            缓存条目数
        """
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        """抛出块类型错误。

        Args:
            blocks: 意外类型的块对象
        """
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """KV 缓存块池管理器。

    负责管理 KVCacheBlocks 的分配、释放和缓存。
    free_block_queue 按驱逐顺序存储空闲块，支持分配、释放和缓存驱逐操作。
    cached_block_hash_to_block 建立块哈希到缓存块的映射，支持通过块哈希查找缓存块。

    Attributes:
        num_gpu_blocks: GPU 上的块总数
        enable_caching: 是否启用前缀缓存
        hash_block_size: 计算块哈希的块大小
        blocks: 所有 KV 缓存块列表
        free_block_queue: 空闲块队列（双向链表）
        cached_block_hash_to_block: 块哈希到块的缓存映射
        null_block: 空块（block_id=0 的占位符）
        enable_kv_cache_events: 是否启用 KV 缓存事件
        kv_event_queue: KV 缓存事件队列
        metrics_collector: 可选的块指标采集器

    Args:
        num_gpu_blocks: GPU 上的块数量
        enable_caching: 是否启用前缀缓存
        hash_block_size: 计算块哈希的块大小
        enable_kv_cache_events: 是否启用 KV 缓存事件
        metrics_collector: 可选的块指标采集器
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        """初始化块池。

        Args:
            num_gpu_blocks: GPU 上的块数量
            enable_caching: 是否启用前缀缓存
            hash_block_size: 计算块哈希的块大小
            enable_kv_cache_events: 是否启用 KV 缓存事件
            metrics_collector: 可选的块指标采集器
        """
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # 所有 kv-cache 块
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # 空闲块队列，构建和操作双向链表
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # 块查找缓存
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # 占位符空块（block_id=0）
        # null_block 的 ref_cnt 不维护，需要特别注意避免释放它
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """获取每个 KV 缓存组中给定块哈希的缓存块。

        对于 kv_cache_group_ids 中的每个组，查找对应的缓存块。
        如果存在重复块，返回缓存中的第一个块。

        Args:
            block_hash: 块的哈希值
            kv_cache_group_ids: KV 缓存组 ID 列表

        Returns:
            如果所有组都存在缓存块则返回块列表，否则返回 None（任一组未命中）
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """缓存请求的完整块用于前缀缓存。

        此函数更新并缓存一组块的块哈希元数据。给定一个请求，它为每个块更新元数据
        并将其添加到 cached_block_hash_to_block 中。块哈希值由 Request 对象在创建时
        以及添加新 token 时立即计算。

        Args:
            request: 要缓存块的请求
            blocks: 请求中的所有块
            num_cached_blocks: 已经缓存的块数量
            num_full_blocks: 在此函数之后应该被缓存的完整块数量
            block_size: 每个块中的 token 数量
            kv_cache_group_id: KV 缓存组 ID
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            # 常见情况
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size 是 hash_block_size 的倍数
            # 当不同 KV 缓存组有不同块大小时会发生这种情况
            assert block_size % self.hash_block_size == 0
            # 使用原始块哈希（hash_block_size 粒度）重新计算 block_size 粒度的块哈希
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # 当启用稀疏注意力（如滑动窗口注意力）或启用前缀缓存的 Mamba 模型时，
            # 一些块可能是空块。跳过空块。
            if blk.is_null:
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # 更新完整块并将其添加到缓存
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # 计算要缓存的块的 token 范围
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # 为每个块单独生成 extra_keys
            # 每个块可能有不同的 extra_keys（例如不同的多模态特征，或仅第一个块的 cache_salt）
            # 跳过空块以匹配 new_hashes 的长度
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """从空闲块池获取新块。

        注意：此函数不检查块缓存。

        Args:
            num_blocks: 要分配的块数量

        Returns:
            新块列表

        Raises:
            ValueError: 如果没有足够的空闲块
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # 为了只迭代列表一次，代码略有重复
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """驱逐缓存块。

        如果一个块在 cached_block_hash_to_block 中被缓存，重置其哈希元数据
        并从缓存中驱逐它。

        Args:
            block: 要驱逐的块

        Returns:
            True 表示块被驱逐，False 表示未驱逐
        """
        # 首先清理指标跟踪以防止泄漏
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # 块没有哈希，不需要驱逐
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # 块不在 cached_block_hash_to_block 中，不需要驱逐
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            # FIXME (Chen): 不确定这里应该返回 hash_value 还是 (hash_value, group_id)
            # 但目前没问题，因为当启用 kv cache event 时禁用了 hybrid kv cache manager，
            # 所以只有一个组
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """触摸块（增加引用计数）。

        触摸一个块会将其引用计数加 1，并可能从空闲队列中移除该块。
        当一个块被具有相同前缀的另一个请求命中时使用此方法。

        Args:
            blocks: 要触摸的块列表
        """
        for block in blocks:
            # ref_cnt=0 表示此块在空闲列表中（即驱逐候选），所以移除它
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """释放一组块。

        块应按驱逐优先级排序，第一个块将最先被驱逐。

        Args:
            ordered_blocks: 按驱逐优先级排序的块列表
        """
        # 物化迭代器以允许多次遍历
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """按块 ID 从前缀缓存中驱逐块。

        仅驱逐当前被缓存的块（具有哈希）。ref_cnt > 0 的块不会从块池中释放，
        仅从前缀缓存哈希表中移除。

        Args:
            block_ids: 要从缓存中驱逐的块 ID 集合

        Raises:
            AssertionError: 如果块 ID 超出范围
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """重置前缀缓存。

        此函数可用于 RLHF 流程中权重更新后使前缀缓存失效，
        或用于基准测试时重置前缀缓存状态。

        Returns:
            True 表示前缀缓存成功重置，False 表示失败（仍有块未释放）
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # null 块总是标记为已使用
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # 移除所有哈希，使新块不会命中缓存
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # 从所有块中移除哈希
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """获取池中空闲块的数量。

        Returns:
            空闲块数量
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """获取 KV 缓存使用率。

        Returns:
            KV 缓存使用率（0.0 到 1.0 之间）
        """
        # 减去 1 以排除 null 块
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """原子性地获取所有事件并清除队列。

        Returns:
            KV 缓存事件列表，如果未启用事件则返回空列表
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events

