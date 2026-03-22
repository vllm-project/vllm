# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""单一类型 KV 缓存管理器模块。

本模块实现了针对不同类型注意力层的 KV 缓存管理器，负责：
- 为每种注意力类型（Full Attention、Sliding Window、Mamba 等）提供专门的缓存管理
- 处理前缀缓存查找和块分配
- 支持滑动窗口、分块局部注意力、Mamba 状态缓存等特性
- 实现块驱逐和复用逻辑

主要类：
- SingleTypeKVCacheManager: 单一类型 KV 缓存管理器抽象基类
- FullAttentionManager: 全注意力管理器
- SlidingWindowManager: 滑动窗口管理器
- ChunkedLocalAttentionManager: 分块局部注意力管理器
- MambaManager: Mamba 状态管理器
- CrossAttentionManager: 交叉注意力管理器
- SinkFullAttentionManager: 带 sink token 的全注意力管理器
"""

import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashWithGroupId,
    KVCacheBlock,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


class SingleTypeKVCacheManager(ABC):
    """单一类型 KV 缓存管理器抽象基类。

    为特定类型的注意力层管理 KV 缓存的抽象基类。
    不同的注意力类型（如全注意力、滑动窗口、Mamba 等）
    需要不同的缓存管理策略，此类定义了统一的接口。

    Attributes:
        block_size: 块大小（token 数量）
        dcp_world_size: 解码上下文并行世界大小
        pcp_world_size: 预填充上下文并行世界大小
        kv_cache_spec: KV 缓存规范
        block_pool: 块池
        enable_caching: 是否启用缓存
        new_block_ids: 新分配的块 ID 列表
        req_to_blocks: 请求 ID 到块的映射
        num_cached_block: 每个请求的缓存块数量
        kv_cache_group_id: KV 缓存组 ID
        _null_block: 空块（用于填充）
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> None:
        """初始化单一类型 KV 缓存管理器。

        Args:
            kv_cache_spec: 此管理器的 kv_cache_spec
            block_pool: 块池
            enable_caching: 是否启用缓存
            kv_cache_group_id: 此管理器所属的 kv 缓存组 ID
            dcp_world_size: 解码上下文并行世界大小
            pcp_world_size: 预填充上下文并行世界大小
        """
        self.block_size = kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size * pcp_world_size > 1:
            self.block_size *= dcp_world_size * pcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self.new_block_ids: list[int] = []

        # 映射请求 ID 到块，用于跟踪每个请求分配的块，
        # 以便在请求完成时释放块。
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: 该请求的缓存块数量}
        # 这用于跟踪每个请求的缓存块数量。
        # 这仅用于跟踪 RUNNING 请求，我们不为被抢占的请求跟踪数据。
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        """获取可驱逐的块数量。

        Args:
            blocks: 块列表

        Returns:
            ref_cnt 为 0 且不是 null_block 的块数量
        """
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        """获取需要为请求分配的块数量。

        Args:
            request_id: 请求 ID
            num_tokens: 需要 slot 的 token 总数（包括已分配的 token）
            new_computed_blocks: 刚命中前缀缓存的新已计算块
            total_computed_tokens: 包括本地和外部已计算 token
            num_tokens_main_model: 主模型（即 spec decode 中的 target model）
                                 的 token 数量。不使用 spec decode 时等于
                                 num_tokens；使用 spec decode 时等于
                                 num_tokens - num_lookahead_tokens

        Returns:
            需要分配的块数量
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # 快速路径：运行中的请求不会有新的前缀缓存命中。
            assert len(new_computed_blocks) == 0
            # 注意：使用推测解码时，请求的块可能为 draft token 分配，
            # 这些 token 后来可能被拒绝。在这种情况下，
            # num_required_blocks 可能小于 num_req_blocks。
            return max(num_required_blocks - num_req_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # 被注意力窗口跳过的完整块数量。
        # 如果没有跳过，这是 0。
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # 我们需要为非跳过后缀分配块。如果窗口内仍有本地计算块，
        # 它们贡献所需容量；否则，跳过的块占主导地位。
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )

        在 `new_computed_blocks` 中，前 `num_skipped_blocks` 个块被跳过；
        `num_req_blocks` 个块可能已经在 `req_to_blocks` 中，
        所以只从 `new_computed_blocks` 中跳过剩余部分。
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # 如果一个已计算块是驱逐候选（在空闲队列中且 ref_cnt == 0），
        # 它将在被分配的请求触摸时从空闲队列中移除，
        # 所以我们必须在空闲容量检查中计算它。
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        return num_new_blocks + num_evictable_blocks

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """将新计算的块添加到请求。

        这涉及三个步骤：
        1. 触摸已计算块以确保它们不会被驱逐
        1.5（可选）对于滑动窗口，用 null_block 填充跳过的块
        2. 添加剩余的计算块
        3. （可选）对于 KV connectors，为外部计算的 token 分配新块（如果有）

        Args:
            request_id: 请求 ID
            new_computed_blocks: 刚命中前缀缓存的新计算块
            num_local_computed_tokens: 本地已计算 token 数量
            num_external_computed_tokens: 外部已计算 token 数量
        """

        if request_id in self.num_cached_block:
            # 快速路径：运行中的请求不会有新的前缀缓存命中。
            # 它不应该有任何新计算的块。
            assert len(new_computed_blocks) == 0
            return

        # 新请求。
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # 当 num_skipped_blocks > len(new_computed_blocks) 时，
            # 所有新计算块都可能被跳过。
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            # 一些外部计算的 token 也可能被跳过。
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        # 触摸已计算块以确保它们不会被驱逐。
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "当前缀缓存禁用时，已计算块应该为空"
            )

        # 用 null_block 填充跳过的块。
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # 添加剩余的计算块。
        req_blocks.extend(new_computed_blocks)
        # 所有缓存命中（包括跳过的 null）都已缓存；标记它们，
        # 这样 cache_blocks() 不会尝试重新缓存已经设置了 block_hash 的块。
        self.num_cached_block[request_id] = len(req_blocks)

        if num_external_computed_tokens > 0:
            # 为外部计算的 token 分配新块。
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """为请求分配新块以提供至少 `num_tokens` 个 token slot。

        Args:
            request_id: 请求 ID
            num_tokens: 需要 slot 的 token 总数（包括已分配的 token）
            num_tokens_main_model: 主模型（即 spec decode 中的 target model）
                                 的 token 数量。不使用 spec decode 时等于
                                 num_tokens；使用 spec decode 时等于
                                 num_tokens - num_lookahead_tokens

        Returns:
            新分配的块列表
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in new_blocks)
            return new_blocks

    def take_new_block_ids(self) -> list[int]:
        """取出并返回自上次调用以来分配的块 ID。

        Returns:
            新块 ID 列表
        """
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """为请求缓存块。

        Args:
            request: 请求
            num_tokens: 需要缓存的 token 总数（包括已缓存的 token）
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        """释放请求的块。

        Args:
            request_id: 请求 ID
        """
        # 默认为 []，以防请求在分配前被释放（中止）。
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # 按相反顺序释放块，以便尾部块先被释放。
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """获取所有分配了 KV 缓存的请求的公共前缀块数量。

        Args:
            running_request_id: 请求 ID

        Returns:
            所有分配了 KV 缓存的请求的公共前缀块数量
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """获取不超过 `max_length` 的最长缓存命中前缀块。

        此前缀应该是 `kv_cache_group_ids` 中所有 kv 缓存组的公共前缀命中。
        如果没有找到缓存命中，返回空列表。
        如果启用了 eagle，删除最后一个匹配的块以强制重新计算最后一个块，
        从而获得 eagle 预测头所需的隐藏状态。
        需要为每种注意力类型定制实现。

        Args:
            block_hashes: 请求的块哈希
            max_length: 缓存命中前缀的最大长度
            kv_cache_group_ids: kv 缓存组 ID 列表
            block_pool: 块池
            kv_cache_spec: kv 缓存规范
            use_eagle: 是否使用 eagle
            alignment_tokens: 返回的缓存命中长度（以 token 为单位）应该是
                           此值（以 token 为单位）的倍数。默认情况下，
                           它应该设置为 block_size
            dcp_world_size: 解码上下文并行的世界大小
            pcp_world_size: 预填充上下文并行的世界大小

        Returns:
            每个 kv 缓存组的已缓存块列表，跳过的块被替换为 null_block。
            返回长度为 `len(kv_cache_group_ids)` 的列表，其中第 i 个元素
            是 `kv_cache_group_ids` 中第 i 个 kv 缓存组的已缓存块列表。
            例如，滑动窗口管理器应该返回一个列表，如
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)])，
            块大小为 4，滑动窗口为 8，len(kv_cache_group_ids) = 1。
        """
        raise NotImplementedError

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """移除不再需要注意力计算的块并释放它们。
        被移除的块应该被 null_block 替换。

        此函数依赖于 `get_num_skipped_tokens`，需要为每种注意力类型
        分别实现。

        Args:
            request_id: 请求 ID
            total_computed_tokens: 已计算 token 总数，包括
                                本地已计算 token 和外部已计算 token
        """
        # 移除在注意力计算期间将被跳过的块。
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            # 这表示所有 token 都在注意力窗口内。
            # 因此我们不需要释放注意力窗口外的任何块。
            # 典型情况是全注意力，我们在请求完成前不释放任何 token。
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` 可能包括尚未分配的 token（例如，当注意力窗口
        # 移动到外部计算的 token 范围内时），所以我们必须限制为
        # 此请求当前存在的块数量。
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        removed_blocks: list[KVCacheBlock] = []
        # 因为块从索引 0 开始，第 num_skipped_block 个块对应索引
        # num_skipped_blocks - 1。
        for i in range(num_skipped_blocks - 1, -1, -1):
            if blocks[i] == self._null_block:
                # 如果块已经是 null_block，它之前的块也应该已经被
                # 之前的调用设置为 null_block。
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """获取将被跳过注意力计算的 token 数量。

        Args:
            num_computed_tokens: 已计算的 token 数量

        Returns:
            将被跳过注意力计算的 token 数量
        """
        # 默认行为是不跳过任何 token。
        return 0

    def new_step_starts(self) -> None:
        """新 step 开始时的回调（默认不执行任何操作）。"""
        # 默认不执行任何操作
        return None


class FullAttentionManager(SingleTypeKVCacheManager):
    """全注意力 KV 缓存管理器。

    用于全注意力层和分块局部注意力层的缓存管理。
    实现简单的前缀缓存查找逻辑。
    """

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """查找最长缓存命中前缀。

        对于全注意力，从左到右遍历块哈希，找到连续的缓存命中前缀。

        Args:
            block_hashes: 请求的块哈希
            max_length: 缓存命中前缀的最大长度
            kv_cache_group_ids: kv 缓存组 ID 列表
            block_pool: 块池
            kv_cache_spec: kv 缓存规范
            use_eagle: 是否使用 eagle
            alignment_tokens: 对齐 token 数量
            dcp_world_size: 解码上下文并行的世界大小
            pcp_world_size: 预填充上下文并行的世界大小

        Returns:
            每个 kv 缓存组的已缓存块列表
        """
        assert isinstance(
            kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
        ), (
            "FullAttentionManager 只能用于全注意力和分块局部注意力组"
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes 是一条块哈希链。如果一个块哈希不在
            # cached_block_hash_to_id 中，后续的块哈希肯定还没有计算。
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            # 如果启用了 eagle，需要删除最后一个匹配的块。
            for computed in computed_blocks:
                computed.pop()
        while (
            block_size != alignment_tokens  # 常见情况的优化
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """获取公共前缀块数量。

        遍历请求的块，统计 ref_cnt 等于请求总数的连续块数量。

        Args:
            running_request_id: 运行中请求的 ID

        Returns:
            公共前缀块数量
        """
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


class SlidingWindowManager(SingleTypeKVCacheManager):
    """滑动窗口 KV 缓存管理器。

    用于滑动窗口注意力层的缓存管理。
    实现了特殊的缓存查找逻辑，从右向左搜索以找到最长的连续缓存命中。

    Attributes:
        sliding_window: 滑动窗口大小
    """

    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        """初始化滑动窗口管理器。

        Args:
            kv_cache_spec: 滑动窗口 kv 缓存规范
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """查找滑动窗口的最长缓存命中前缀。

        从右向左搜索，找到最长的连续缓存命中。
        滑动窗口需要连续的缓存命中才能有效。

        Args:
            block_hashes: 请求的块哈希
            max_length: 缓存命中前缀的最大长度
            kv_cache_group_ids: kv 缓存组 ID 列表
            block_pool: 块池
            kv_cache_spec: kv 缓存规范
            use_eagle: 是否使用 eagle
            alignment_tokens: 对齐 token 数量
            dcp_world_size: 解码上下文并行的世界大小
            pcp_world_size: 预填充上下文并行的世界大小

        Returns:
            每个 kv 缓存组的已缓存块列表，窗口外的块用 null_block 填充
        """
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager 只能用于滑动窗口组"
        )
        assert dcp_world_size == 1, "DCP 目前不支持滑动窗口注意力"
        assert pcp_world_size == 1, "PCP 目前不支持滑动窗口注意力"

        # 前缀缓存命中所需的连续块数量。
        # -1 是因为输入 token 本身也包含在窗口内
        sliding_window_contiguous_blocks = cdiv(
            kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size
        )
        if use_eagle:
            # 如果启用了 eagle，需要删除最后一个匹配的块。
            # 对于滑动窗口层，我们通过将前缀缓存命中所需的连续块数量
            # 增加 1 并删除最后一个匹配的块来实现。
            sliding_window_contiguous_blocks += 1

        # TODO: 当缓存未命中时减少 i 滑动 sliding_window_contiguous_blocks，
        # 将时间复杂度从 O(max_num_blocks) 优化到
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks)，
        # 这对低缓存命中率场景有益。
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        # 从右向左搜索，找到匹配后提前停止。
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # 如果块与 `alignment_tokens` 不对齐，跳过前缀匹配检查。
                if (
                    num_contiguous_blocks == 0
                    and block_size != alignment_tokens  # 常见情况的优化
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                # 将已缓存块添加到已计算块中。
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # 删除尾部块。
                    # 例如，[NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # 当 sliding_window_contiguous_blocks=2 时。
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # 即使 `num_contiguous_blocks < sliding_window_contiguous_blocks`，
            # 前 `num_contiguous_blocks` 个也是缓存命中。
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            while (
                block_size != alignment_tokens  # 常见情况的优化
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        if use_eagle and computed_blocks[0]:
            assert kv_cache_spec.block_size == alignment_tokens, (
                "aligned_length 目前与 eagle 不兼容"
            )
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """获取将被跳过注意力计算的 token 数量。

        对于滑动窗口，这对应于当前滑动窗口之前的 token。

        示例：
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ 下一个要计算的 token
                               |-----------| 下一个 token 的滑动窗口
                  |--skipped---|

        当前窗口包含 token 4~7。Token 0~3 将被跳过注意力计算，
        因为它们在滑动窗口外。因此，get_num_skipped_tokens(7) == 4。

        Args:
            num_computed_tokens: 已计算的 token 数量

        Returns:
            将被跳过注意力计算的 token 数量
        """
        return max(0, num_computed_tokens - self.sliding_window + 1)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """获取公共前缀块数量。

        注意 (Chen): 对于滑动窗口层，前缀块是 null_block。
        所以像 FullAttentionManager 那样计算 ref_cnt 是不正确的。
        这里返回 0 以保证正确性。需要在未来支持级联注意力 + 滑动窗口。

        Args:
            running_request_id: 运行中请求的 ID

        Returns:
            0（滑动窗口不支持公共前缀块）
        """
        return 0


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    """分块局部注意力 KV 缓存管理器。

    用于分块局部注意力层的缓存管理。
    只有当前分块内的 token 需要注意力计算，之前的分块用 null_block 标记。

    Attributes:
        attention_chunk_size: 注意力分块大小
    """

    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None:
        """初始化分块局部注意力管理器。

        Args:
            kv_cache_spec: 分块局部注意力 kv 缓存规范
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(kv_cache_spec, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """查找分块局部注意力的最长缓存命中前缀。

        对于分块局部注意力，我们需要找到不超过 `max_length` 的最长缓存命中前缀。
        此前缀应该是 `kv_cache_group_ids` 中所有 kv 缓存组的公共前缀命中。
        如果没有找到缓存命中，返回空列表。
        注意：如果整个块在局部窗口外，我们将其标记为已计算，并将块设置为 null。

        示例：

        1. 注意力分块大小为 8，块大小为 4，最大长度为 15
           对于第 15 个 token（从 0 开始索引），第 8-14 个 token 在窗口内（需要查找），
           第 0-7 个 token 不在窗口内，所以它们已经被标记为已计算。
           我们检查完整的 block3（第 8-11 个 token），假设 block 3 命中，
           我们将返回 [null, null, block 3]，否则返回 [null, null]

        2. 注意力分块大小为 8，块大小为 4，最大长度为 16
           对于第 16 个 token（从 0 开始索引），第 0-15 个 token 不在窗口内，
           所以它们已经被标记为已计算。
           我们返回 4 个块 [null, null, null, null]

        Args:
            block_hashes: 请求的块哈希
            max_length: 缓存命中前缀的最大长度
            kv_cache_group_ids: kv 缓存组 ID 列表
            block_pool: 块池
            kv_cache_spec: kv 缓存规范
            use_eagle: 是否使用 eagle
            alignment_tokens: 返回的缓存命中长度（以 token 为单位）应该是
                           此值（以 token 为单位）的倍数
            dcp_world_size: 解码上下文并行的世界大小
            pcp_world_size: 预填充上下文并行的世界大小

        Returns:
            已缓存块列表
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager 只能用于分块局部注意力组"
        )
        assert use_eagle is False, (
            "混合 KV 缓存不支持 eagle + 分块局部注意力"
        )
        assert dcp_world_size == 1, "DCP 目前不支持分块局部注意力"
        assert pcp_world_size == 1, "PCP 目前不支持分块局部注意力"
        assert kv_cache_spec.block_size == alignment_tokens, (
            "具有不同块大小的 KV 缓存组目前与分块局部注意力不兼容"
        )
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (
                max_length
                // kv_cache_spec.attention_chunk_size
                * kv_cache_spec.attention_chunk_size
            )
        else:
            local_attention_start_idx = 0
        # 我们将窗口外的块标记为已计算，使用 null_block，
        # 并根据缓存查找结果标记窗口内的块
        # [null] [null] ... [null] [hit block 1 (包含最后窗口的第一个块)]
        # [hit block 2] ... [hit block x]
        local_attention_start_block_idx = (
            local_attention_start_idx // kv_cache_spec.block_size
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * local_attention_start_block_idx
            for _ in range(len(kv_cache_group_ids))
        )
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """获取将被跳过注意力计算的 token 数量。

        对于分块局部注意力，这对应于当前分块左侧的 token。

        示例 1：
        chunk size = 8, num_computed_tokens = 13
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | ----- computed ---------------|
                                                  ^^ 下一个要计算的 token
                                   |----------------| <-- 下一个 token 的注意力窗口
                 |--- skipped -----|
        输出：get_num_skipped_tokens(13) == 8

        示例 2：
        chunk size = 8, num_computed_tokens = 8
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | --- computed ---|
                                     ^ 下一个要计算的 token
                                   |--| <-- 下一个 token 的注意力窗口
                 | --- skipped ----|
        输出：get_num_skipped_tokens(8) == 8

        示例 3：
        chunk size = 8, num_computed_tokens = 7
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 |---computed---|
                                 ^ 下一个要计算的 token
                 |-----------------| <-- 下一个 token 的注意力窗口
                 没有 token 应该被跳过。
        输出：get_num_skipped_tokens(7) == 0

        Args:
            num_computed_tokens: 已计算的 token 数量

        Returns:
            将被跳过注意力计算的 token 数量
        """
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """获取公共前缀块数量。

        分块局部注意力不支持级联注意力。

        Args:
            running_request_id: 运行中请求的 ID

        Returns:
            0（分块局部注意力不支持公共前缀块）
        """
        return 0


class MambaManager(SingleTypeKVCacheManager):
    """Mamba KV 缓存管理器。

    用于 Mamba 层（线性注意力/状态空间模型）的缓存管理。
    实现了特殊的缓存逻辑，因为 Mamba 只需要保留最后一个计算 token 的状态。

    Attributes:
        cached_blocks_this_step: 当前 step 中缓存的块哈希集合
        mamba_cache_mode: Mamba 缓存模式
        num_speculative_blocks: 推测解码所需的额外块数量
        last_state_block_idx: 每个请求的上一步分配的块索引（align 模式）
        _allocated_block_reqs: 已分配块的请求集合（align 模式）
    """

    def __init__(
        self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        """初始化 Mamba 管理器。

        Args:
            kv_cache_spec: Mamba kv 缓存规范
            block_pool: 块池
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.cached_blocks_this_step: set[BlockHashWithGroupId] = set()
        self.mamba_cache_mode = kv_cache_spec.mamba_cache_mode
        self.num_speculative_blocks: int = kv_cache_spec.num_speculative_blocks
        if self.mamba_cache_mode == "align":
            # 映射请求 ID 到上一步分配的块索引
            self.last_state_block_idx: dict[str, int] = {}
            # 已分配块的请求集合
            self._allocated_block_reqs: set[str] = set()

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """查找 Mamba 的最长缓存命中前缀。

        从右向左搜索，找到最后一个匹配的块（Mamba 只需要最后一个状态）。

        Args:
            block_hashes: 请求的块哈希
            max_length: 缓存命中前缀的最大长度
            kv_cache_group_ids: kv 缓存组 ID 列表
            block_pool: 块池
            kv_cache_spec: kv 缓存规范
            use_eagle: 是否使用 eagle
            alignment_tokens: 对齐 token 数量
            dcp_world_size: 解码上下文并行的世界大小
            pcp_world_size: 预填充上下文并行的世界大小

        Returns:
            每个 kv 缓存组的已缓存块列表
        """
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager 只能用于 Mamba 组"
        )
        assert dcp_world_size == 1, "DCP 目前不支持 Mamba"
        assert pcp_world_size == 1, "PCP 目前不支持 Mamba"
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )

        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        # 从右向左搜索，找到匹配后提前停止。
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # 当启用 Mamba 前缀缓存时，`block_size` 将在全注意力层和
                # Mamba 层之间对齐，以确保前缀命中长度在块边界对齐
                if (
                    block_size != alignment_tokens  # 常见情况的优化
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # 命中长度逻辑后续假设：
                    #   hit_length = len(hit_blocks_other_attn[0])
                    #                * self.other_block_size
                    # 所以我们在开头插入 dummy 块：
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                break  # 我们只需要最后一个匹配 - 提前停止

        return computed_blocks

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        """移除并释放不再需要的 Mamba 状态块。

        注意 (tdoublep) 使用异步调度时，num_computed_tokens 可能包含
        上一步的 draft token，这些 token 可能后来被拒绝。
        这可能会让我们认为我们实际上比实际更靠后，
        所以让我们假设所有 token 都被拒绝，以免释放我们实际需要的块。

        Args:
            request_id: 请求 ID
            num_computed_tokens: 已计算 token 数量
        """
        assert isinstance(self.kv_cache_spec, MambaSpec)

        # 注意 (tdoublep) 使用异步调度时，num_computed_tokens 可能包含
        # 上一步的 draft token，这些 token 可能后来被拒绝。
        # 这可能会让我们认为我们实际上比实际更靠后，
        # 所以让我们假设所有 token 都被拒绝，以免释放我们实际需要的块。
        num_computed_tokens = max(0, num_computed_tokens - self.num_speculative_blocks)

        super().remove_skipped_blocks(request_id, num_computed_tokens)
        if self.mamba_cache_mode == "align":
            # `last_state_block_idx` 指的是两步前分配的块索引。
            # 上一步分配的块用于将 Mamba 状态复制到当前步骤分配的块中；
            # 更早的块不再需要，应该在这里释放。
            last_state_block_idx = self.last_state_block_idx.get(request_id)
            # 预填充期间分配的块可能是非连续的。使用
            # `last_state_block_idx` 来释放适当的块并将其替换为 null_block。
            if (
                last_state_block_idx is not None
                and last_state_block_idx
                < cdiv(num_computed_tokens, self.block_size) - 1
            ):
                blocks = self.req_to_blocks[request_id]
                if blocks[last_state_block_idx] != self._null_block:
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """获取公共前缀块数量。

        Mamba 不支持级联注意力。

        Args:
            running_request_id: 运行中请求的 ID

        Returns:
            0（Mamba 不支持公共前缀块）
        """
        return 0

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        """获取需要为 Mamba 请求分配的块数量。

        Args:
            request_id: 请求 ID
            num_tokens: 需要 slot 的 token 总数
            new_computed_blocks: 新计算的块
            total_computed_tokens: 总已计算 token 数量
            num_tokens_main_model: 主模型的 token 数量

        Returns:
            需要分配的块数量
        """
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if (
            len(new_computed_blocks) > 0
            and new_computed_blocks[-1].block_hash in self.cached_blocks_this_step
        ):
            # Mamba 不能依赖当前 step 中其他请求生成的块
            # 为了推迟到下一步，我们返回 num_gpu_blocks + 1，
            # 这样 kv_cache_manager 会认为现在没有足够的块可分配，
            # 而不会在当前 step 中调度它。
            return self.block_pool.num_gpu_blocks + 1
        if self.mamba_cache_mode != "align":
            # 为线性注意力的推测解码（MTP/EAGLE）分配额外的
            # `num_speculative_blocks` 个块。
            if self.num_speculative_blocks > 0:
                num_tokens += (
                    self.kv_cache_spec.block_size * self.num_speculative_blocks
                )
            return super().get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_tokens_main_model,
            )
        else:
            # 在 align 模式下，我们不为 lookahead token 分配块，因为如果
            # x * block_size 个 token 被调度，num_tokens 是
            # x * block_size + num_lookahead_tokens，这会破坏对齐。
            # 我们可以忽略 lookahead token，因为当前的 draft 模型没有
            # mamba 层。
            num_tokens = num_tokens_main_model

            # 注意 (tdouble): 这高估了我们需要的块数量，因为
            # num_tokens 可能包含后来被拒绝的 draft token。
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            num_new_blocks = (
                num_required_blocks
                - len(new_computed_blocks)
                - len(self.req_to_blocks[request_id])
            )
            if num_new_blocks > 0:
                if request_id in self._allocated_block_reqs:
                    # 旧请求。最多需要 1 个额外的块，因为我们可以重用
                    # 上一步的推测块。
                    num_new_blocks = 1
                else:
                    # 第一次预填充。分配 1 个块用于运行状态和推测块。
                    num_new_blocks = 1 + self.num_speculative_blocks

            num_evictable_computed_blocks = self._get_num_evictable_blocks(
                new_computed_blocks
            )
            return num_new_blocks + num_evictable_computed_blocks

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """为 Mamba 请求分配新块。

        Args:
            request_id: 请求 ID
            num_tokens: 需要 slot 的 token 总数
            num_tokens_main_model: 主模型的 token 数量

        Returns:
            新分配的块列表
        """
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.mamba_cache_mode != "align":
            # 为线性注意力的推测解码（MTP/EAGLE）分配额外的
            # `num_speculative_blocks` 个块。
            if self.num_speculative_blocks > 0:
                num_tokens += self.block_size * self.num_speculative_blocks
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )
        else:
            # 在 align 模式下，我们不为 lookahead token 分配块，因为如果
            # x * block_size 个 token 被调度，num_tokens 是
            # x * block_size + num_lookahead_tokens，这会破坏对齐。
            # 我们可以忽略 lookahead token，因为当前的 draft 模型没有
            # mamba 层。
            num_tokens = num_tokens_main_model
            req_blocks: list[KVCacheBlock] = self.req_to_blocks[request_id]
            # 注意 (tdouble): 这高估了我们需要的块数量，因为
            # num_tokens 可能包含后来被拒绝的 draft token。
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            if num_required_blocks == len(req_blocks):
                return []
            else:
                assert num_required_blocks > len(req_blocks), (
                    f"num_required_blocks {num_required_blocks} < "
                    f"len(req_blocks) {len(req_blocks)}"
                )
                prev_block_len = len(req_blocks)
                blocks_allocated = request_id in self._allocated_block_reqs
                # 记录最后一个状态块
                if blocks_allocated:
                    # 我们总是将运行状态保存在最后一个
                    # (1 + num_speculative_blocks) 块
                    self.last_state_block_idx[request_id] = (
                        prev_block_len - 1 - self.num_speculative_blocks
                    )
                elif prev_block_len > 0:
                    # 当新请求命中前缀缓存时，最后一个块保存命中的状态。
                    self.last_state_block_idx[request_id] = prev_block_len - 1

                num_skipped_blocks = (
                    num_required_blocks - self.num_speculative_blocks - 1
                )
                # null 块
                if prev_block_len < num_skipped_blocks:
                    req_blocks.extend(
                        [
                            self._null_block
                            for _ in range(prev_block_len, num_skipped_blocks)
                        ]
                    )

                if blocks_allocated:
                    # 在这一步重用之前的推测块
                    for block_idx in range(
                        prev_block_len - self.num_speculative_blocks, prev_block_len
                    ):
                        if block_idx < num_skipped_blocks:
                            req_blocks.append(req_blocks[block_idx])
                            req_blocks[block_idx] = self._null_block
                        else:
                            break
                num_new_blocks = num_required_blocks - len(req_blocks)
                if blocks_allocated:
                    assert num_new_blocks <= 1
                else:
                    assert num_new_blocks <= self.num_speculative_blocks + 1
                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
                req_blocks.extend(new_blocks)
                self._allocated_block_reqs.add(request_id)
                return req_blocks[prev_block_len:]

    def free(self, request_id: str) -> None:
        """释放 Mamba 请求的块。

        Args:
            request_id: 请求 ID
        """
        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
        super().free(request_id)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """获取将被跳过注意力计算的 token 数量。

        Mamba 只需要保留最后一个计算 token 的状态，
        所以我们返回 num_computed_tokens - 1。

        Args:
            num_computed_tokens: 已计算的 token 数量

        Returns:
            将被跳过注意力计算的 token 数量
        """
        return num_computed_tokens - 1

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """为 Mamba 请求缓存块。

        跟踪当前 step 中缓存的块哈希，以防止在同一 step 中重复使用。

        Args:
            request: 请求
            num_tokens: 需要缓存的 token 总数
        """
        num_cached_blocks_before = self.num_cached_block.get(request.request_id, 0)
        super().cache_blocks(request, num_tokens)
        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)
        if num_cached_blocks_after > num_cached_blocks_before:
            for block in self.req_to_blocks[request.request_id][
                num_cached_blocks_before:num_cached_blocks_after
            ]:
                if block.is_null:
                    continue
                assert block.block_hash is not None
                self.cached_blocks_this_step.add(block.block_hash)

    def new_step_starts(self) -> None:
        """新 step 开始时清除缓存块集合。"""
        self.cached_blocks_this_step.clear()


class CrossAttentionManager(SingleTypeKVCacheManager):
    """交叉注意力 KV 缓存管理器。

    用于编码器 - 解码器模型中的交叉注意力 KV 缓存。
    不实现前缀缓存，因为编码器状态是每个请求唯一的。
    """

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """分配新计算的块。

        我们不缓存交叉注意力块以在请求之间共享，
        所以 `new_computed_blocks` 应该总是为空。

        Args:
            request_id: 请求 ID
            new_computed_blocks: 新计算的块
            num_local_computed_tokens: 本地已计算 token 数量
            num_external_computed_tokens: 外部已计算 token 数量
        """
        # 我们不缓存交叉注意力块以在请求之间共享，
        # 所以 `new_computed_blocks` 应该总是为空。
        assert len(new_computed_blocks) == 0

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """缓存块（交叉注意力不支持）。

        Args:
            request: 请求
            num_tokens: 需要缓存的 token 总数

        Raises:
            ValueError: 总是抛出，因为前缀缓存被禁用
        """
        # 我们不缓存交叉注意力块以在请求之间共享，
        # 所以此方法不相关。
        raise ValueError("不应调用，因为前缀缓存被禁用。")

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """获取公共前缀块数量。

        交叉注意力块包含请求特定的编码器状态，
        不在不同请求之间共享。

        Args:
            running_request_id: 运行中请求的 ID

        Returns:
            0（交叉注意力不支持公共前缀块）
        """
        # 交叉注意力块包含请求特定的编码器状态，
        # 不在不同请求之间共享
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """查找最长缓存命中前缀（交叉注意力不支持）。

        交叉注意力不从前缀缓存中受益，因为：
        1. 编码器状态是每个请求唯一的（不同的音频/图像输入）
        2. 编码器状态每个请求只计算一次，不是增量计算
        3. 不同多模态输入之间没有可重用的前缀

        Args:
            block_hashes: 请求的块哈希
            max_length: 缓存命中前缀的最大长度
            kv_cache_group_ids: kv 缓存组 ID 列表
            block_pool: 块池
            kv_cache_spec: kv 缓存规范
            use_eagle: 是否使用 eagle
            alignment_tokens: 对齐 token 数量
            dcp_world_size: 解码上下文并行的世界大小
            pcp_world_size: 预填充上下文并行的世界大小

        Returns:
            抛出 NotImplementedError
        """
        assert isinstance(kv_cache_spec, CrossAttentionSpec), (
            "CrossAttentionManager 只能用于交叉注意力组"
        )
        # 交叉注意力不从前缀缓存中受益，因为：
        # 1. 编码器状态是每个请求唯一的（不同的音频/图像输入）
        # 2. 编码器状态每个请求只计算一次，不是增量计算
        # 3. 不同多模态输入之间没有可重用的前缀
        # 返回空块以表示没有缓存命中
        raise NotImplementedError("CrossAttentionManager 不支持缓存")


class SinkFullAttentionManager(FullAttentionManager):
    """带 sink token 的全注意力 KV 缓存管理器。

    用于具有 sink token 的全注意力层。
    Sink token 是始终保留在缓存中的特殊 token，
    用于处理长序列中的局部注意力。

    Attributes:
        sink_blocks: sink token 的专用块
    """

    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        """初始化带 sink token 的全注意力管理器。

        Args:
            kv_cache_spec: sink 全注意力 kv 缓存规范
            block_pool: 块池
            enable_caching: 是否启用缓存
            kv_cache_group_id: kv 缓存组 ID
            dcp_world_size: 解码上下文并行的世界大小
            pcp_world_size: 预填充上下文并行的世界大小
        """
        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            dcp_world_size,
            pcp_world_size,
        )
        sink_len = kv_cache_spec.sink_len
        assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)


# 规范到管理器的映射表
spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
    SinkFullAttentionSpec: SinkFullAttentionManager,
}


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec, **kwargs
) -> SingleTypeKVCacheManager:
    """根据 KV 缓存规范获取对应的管理器。

    Args:
        kv_cache_spec: KV 缓存规范
        **kwargs: 传递给管理器的其他参数

    Returns:
        对应的 KV 缓存管理器实例
    """
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
