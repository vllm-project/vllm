# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 缓存协调器模块。

本模块实现了 KV 缓存协调器功能，负责协调多个 KV 缓存组的管理。
主要功能包括：
- 协调不同类型的 KV 缓存管理器（FullAttention、SlidingWindow、Mamba 等）
- 处理多 KV 缓存组的最长缓存命中查找
- 支持无前缀缓存、单一缓存组和混合缓存组三种场景

主要类：
- KVCacheCoordinator: KV 缓存协调器抽象基类
- KVCacheCoordinatorNoPrefixCache: 无前缀缓存协调器
- UnitaryKVCacheCoordinator: 单一 KV 缓存组协调器
- HybridKVCacheCoordinator: 混合 KV 缓存组协调器
"""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import lcm

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.request import Request


class KVCacheCoordinator(ABC):
    """KV 缓存协调器（抽象基类）。

    协调不同 KV 缓存组的缓存管理，包括：
    - 为请求分配和释放块
    - 缓存块用于前缀缓存
    - 查找最长缓存命中
    - 追踪常见前缀块

    Attributes:
        kv_cache_config: KV 缓存配置
        max_model_len: 最大模型长度
        enable_caching: 是否启用前缀缓存
        block_pool: 块池管理器
        use_eagle: 是否使用 EAGLE
        single_type_managers: 单一类型 KV 缓存管理器元组
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        """初始化 KV 缓存协调器。

        Args:
            kv_cache_config: KV 缓存配置
            max_model_len: 最大模型长度
            use_eagle: 是否使用 EAGLE
            enable_caching: 是否启用前缀缓存
            enable_kv_cache_events: 是否启用 KV 缓存事件
            dcp_world_size: Decode 上下文并行世界大小
            pcp_world_size: Prefill 上下文并行世界大小
            hash_block_size: 计算块哈希的块大小
            metrics_collector: 可选的指标采集器
        """
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        self.block_pool = BlockPool(
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )

        # 如果启用 EAGLE，find_longest_cache_hit 需要特殊处理
        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        """获取请求需要分配的块数量。

        Args:
            request_id: 请求 ID
            num_tokens: 需要槽位的 token 总数（包括已分配的 token）
            new_computed_blocks: 刚刚命中前缀缓存的新计算块
            num_encoder_tokens: 用于分配交叉注意力块的编码器 token 数量
            total_computed_tokens: 包括本地和外部计算 token
            num_tokens_main_model: 主模型（即 spec decode 中的目标模型）的 token 数量。
                不使用 spec decode 时为 num_tokens；
                使用 spec decode 时为 num_tokens - num_lookahead_tokens

        Returns:
            要分配的块数量
        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # 对于交叉注意力，基于编码器输入 token 数量进行单次静态块分配
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id, num_encoder_tokens, [], 0, num_encoder_tokens
                )
            else:
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_tokens_main_model,
                )
        return num_blocks_to_allocate

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """将新计算块添加到请求中。

        可选择为外部计算 token 分配新块（如果有）。

        Args:
            request_id: 请求 ID
            new_computed_blocks: 刚刚命中前缀缓存的新计算块
            num_local_computed_tokens: 本地计算 token 数量
            num_external_computed_tokens: 外部计算 token 数量
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.allocate_new_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """为请求分配新块以提供至少 num_tokens 个 token 槽位。

        Args:
            request_id: 请求 ID
            num_tokens: 需要槽位的 token 总数（包括已分配的 token）
            num_tokens_main_model: 主模型（即 spec decode 中的目标模型）的 token 数量
            num_encoder_tokens: 用于分配交叉注意力块的编码器 token 数量（默认 0）

        Returns:
            新分配的块元组
        """
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
                num_tokens_main_model,
            )
            for manager in self.single_type_managers
        )

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """缓存请求的块。

        Args:
            request: 请求
            num_computed_tokens: 需要缓存的 token 总数（包括已缓存的 token）
        """
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """释放请求的块。

        Args:
            request_id: 请求 ID
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """获取每个 KV 缓存组的所有请求的常见前缀块数量。

        Args:
            running_request_id: 任意运行中请求的 ID，用于识别常见前缀块

        Returns:
            每个 KV 缓存组的常见前缀块数量列表
        """
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """从块中移除不再需要的块并用 null_block 替换。

        Args:
            request_id: 请求 ID
            total_computed_tokens: 计算 token 总数，包括本地和外部计算 token
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, total_computed_tokens)

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """获取请求的块。

        Args:
            request_id: 请求 ID

        Returns:
            每个管理器的块列表元组
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """查找最长缓存命中。

        Args:
            block_hashes: 请求的块哈希列表
            max_cache_hit_length: 缓存命中的最大长度

        Returns:
            包含每个管理器的缓存命中块元组和命中 token 数量的元组
        """
        pass

    def new_step_starts(self) -> None:
        """新 step 开始时的回调。"""
        for manager in self.single_type_managers:
            manager.new_step_starts()


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """无前缀缓存的 KV 缓存协调器。

    当前缀缓存被禁用或不支持时使用。
    与 UnitaryKVCacheCoordinator 和 HybridKVCacheCoordinator 相比，
    支持任意数量的 KV 缓存组（包括 0 个组）。
    不实现任何与前缀缓存相关的功能。
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        """初始化无前缀缓存协调器。

        Args:
            kv_cache_config: KV 缓存配置
            max_model_len: 最大模型长度
            use_eagle: 是否使用 EAGLE
            enable_kv_cache_events: 是否启用 KV 缓存事件
            dcp_world_size: Decode 上下文并行世界大小
            pcp_world_size: Prefill 上下文并行世界大小
            hash_block_size: 计算块哈希的块大小
            metrics_collector: 可选的指标采集器
        """
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            False,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """获取常见前缀块数量（无前缀缓存时返回全 0）。

        Args:
            running_request_id: 运行中请求 ID

        Returns:
            全 0 列表，长度为管理器数量
        """
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """查找最长缓存命中（无前缀缓存时返回空结果）。

        Args:
            block_hashes: 请求的块哈希列表
            max_cache_hit_length: 缓存命中最大长度

        Returns:
            空块元组和 0
        """
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """单一 KV 缓存组协调器。

    用于只有一个 KV 缓存组的模型，即所有注意力层使用相同类型注意力的模型，
    例如所有注意力层都使用 full attention 或都使用 sliding window attention。
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        """初始化单一 KV 缓存组协调器。

        Args:
            kv_cache_config: KV 缓存配置
            max_model_len: 最大模型长度
            use_eagle: 是否使用 EAGLE
            enable_caching: 是否启用前缀缓存
            enable_kv_cache_events: 是否启用 KV 缓存事件
            dcp_world_size: Decode 上下文并行世界大小
            pcp_world_size: Prefill 上下文并行世界大小
            hash_block_size: 计算块哈希的块大小
            metrics_collector: 可选的指标采集器
        """
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        if pcp_world_size > 1:
            self.block_size *= pcp_world_size
        # 对于仅使用 Mamba 的模型，当前缀缓存被禁用时 block_size 设置为 max_model_len，
        # 并且跳过 hash_block_size 验证
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """查找最长缓存命中（单一缓存组）。

        Args:
            block_hashes: 请求的块哈希列表
            max_cache_hit_length: 缓存命中最大长度

        Returns:
            包含缓存命中块元组和命中 token 数量的元组
        """
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """混合 KV 缓存组协调器。

    用于具有多种 KV 缓存类型的混合模型，因此有多个 KV 缓存组。
    实现了迭代定点算法来查找最长缓存命中。
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        """初始化混合 KV 缓存组协调器。

        Args:
            kv_cache_config: KV 缓存配置
            max_model_len: 最大模型长度
            use_eagle: 是否使用 EAGLE
            enable_caching: 是否启用前缀缓存
            enable_kv_cache_events: 是否启用 KV 缓存事件
            dcp_world_size: Decode 上下文并行世界大小
            pcp_world_size: Prefill 上下文并行世界大小
            hash_block_size: 计算块哈希的块大小
            metrics_collector: 可选的指标采集器
        """
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        # hash_block_size: 用于计算块哈希的块大小
        # 实际块大小通常等于 hash_block_size，但在不同 KV 缓存组有不同块大小的情况下，
        # 实际块大小可以是 hash_block_size 的倍数
        self.hash_block_size = hash_block_size
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0
            for g in kv_cache_config.kv_cache_groups
        ), "block_size must be divisible by hash_block_size"
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """按 spec 类型分组 KV 缓存组以进行高效批量处理。

        在缓存命中查找期间，将具有相同 spec 的 KV 缓存组分组在一起以提高效率。
        """
        attention_groups: list[
            tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]
        ] = []

        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # 尝试查找具有相同 spec 的现有组
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        assert len(attention_groups) > 1, (
            "HybridKVCacheCoordinator requires at least two attention groups."
        )

        # 将 full attention 放在前面：其从左到右的高效扫描提供了更紧的初始边界，
        # 减少了后续组的工作量
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )

        # 所有注意力类型块大小的最小公倍数
        # 缓存命中长度必须是块大小最小公倍数的倍数，以确保缓存命中长度
        # 是每种注意力类型块大小的倍数。因为目前不支持部分块缓存命中，所以需要这个约束
        block_sizes = [spec.block_size for spec, _, _ in attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """使用迭代定点算法查找最长缓存命中。

        每种注意力类型要么接受当前候选长度，要么减少它。如果任何类型减少了长度，
        则重新对所有类型进行检查。由于长度单调递减且有下界 0，所以算法会收敛。

        Args:
            block_hashes: 请求的块哈希列表
            max_cache_hit_length: 缓存命中的最大长度

        Returns:
            包含以下内容的元组：
                - 每个单一类型管理器的缓存命中块元组
                - 最长缓存命中的 token 数量
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            """获取指定 KV 缓存规范的块哈希列表。"""
            if kv_cache_spec.block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, kv_cache_spec.block_size
            )

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        # 简单混合（1 个 full attn + 1 个其他）：一次迭代就足够了
        # 如果存在 full attention，它总是排在第一位。这避免了 EAGLE 丢弃
        # 被多次应用于非 full-attn 组
        # FIXME (yifan): 然而，对于具有多个注意力组的复杂混合模型，
        # 我们仍然存在 EAGLE 螺旋块丢弃问题。参见 issue 讨论：
        # https://github.com/vllm-project/vllm/issues/32802
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )

        while True:
            curr_hit_length = hit_length

            for spec, group_ids, manager_cls in self.attention_groups:
                is_full_attn = isinstance(spec, FullAttentionSpec)

                # Full attention: 重用缓存的块（向下闭合属性）
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if is_full_attn and cached_blocks is not None:
                    # 对于 full attention，我们只需要计算一次缓存命中长度
                    # 从第二次迭代开始，如果 curr_hit_length 被其他组减少，
                    # 我们可以简单地保留上一次迭代的前 (curr_hit_length // block_size) 个块
                    num_blocks = curr_hit_length // spec.block_size
                    curr_hit_length = num_blocks * spec.block_size
                else:
                    hit_blocks = manager_cls.find_longest_cache_hit(
                        block_hashes=_get_block_hashes(spec),
                        max_length=curr_hit_length,
                        kv_cache_group_ids=group_ids,
                        block_pool=self.block_pool,
                        kv_cache_spec=spec,
                        use_eagle=self.use_eagle,
                        alignment_tokens=self.lcm_block_size,
                    )
                    curr_hit_length = len(hit_blocks[0]) * spec.block_size
                    for group_id, blocks in zip(group_ids, hit_blocks):
                        hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            # 简单混合：一次迭代后退出
            if is_simple_hybrid:
                break

        # 将 full attention 块截断到最终的 hit_length（如果存在）
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            num_blocks = hit_length // spec.block_size
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        return tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        ), hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    """获取 KV 缓存协调器的工厂函数。

    根据配置返回适当类型的 KVCacheCoordinator：
    - 如果不启用缓存，返回 KVCacheCoordinatorNoPrefixCache
    - 如果只有一个 KV 缓存组，返回 UnitaryKVCacheCoordinator
    - 否则返回 HybridKVCacheCoordinator

    Args:
        kv_cache_config: KV 缓存配置
        max_model_len: 最大模型长度
        use_eagle: 是否使用 EAGLE
        enable_caching: 是否启用前缀缓存
        enable_kv_cache_events: 是否启用 KV 缓存事件
        dcp_world_size: Decode 上下文并行世界大小
        pcp_world_size: Prefill 上下文并行世界大小
        hash_block_size: 计算块哈希的块大小
        metrics_collector: 可选的指标采集器

    Returns:
        适当类型的 KVCacheCoordinator 实例
    """
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
    )
