# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 缓存管理器模块。

本模块实现了 vLLM V1 的 KV 缓存管理核心功能，负责：
- 管理请求的 KV 缓存块分配和释放
- 实现前缀缓存查找和复用
- 处理滑动窗口、推测解码等高级特性
- 提供统一的缓存管理接口

主要类：
- KVCacheBlocks: KV 缓存块分配结果的封装
- KVCacheManager: KV 缓存管理器
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """KV 缓存块分配结果封装类。

    作为 Scheduler 和 KVCacheManager 之间的接口，
    隐藏 KVCacheManager 的内部数据结构。

    Attributes:
        blocks: 块元组，blocks[i][j] 表示第 i 个 kv_cache_group
                的第 j 个 token 块
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` 指的是第 i 个 kv_cache_group 和第 j 个 token 块。
    我们不使用 token 块作为外层维度，因为这假设所有 kv_cache_group
    都有相同的块数量，目前这是正确的，但如果我们想在未来给不同的
    kv_cache_group 不同的 block_size，这个假设就会被打破。

    每个单一类型的 KVCacheBlocks 可以表示为：
    - list[KVCacheBlock] 表示多于一个 KVCacheBlock
    - 空元组表示没有 KVCacheBlock 的请求（KVCacheManager 中有预计算的
      KVCacheBlocks 以避免 GC 开销）
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """添加两个 KVCacheBlocks 实例。

        Args:
            other: 另一个 KVCacheBlocks 实例

        Returns:
            合并后的 KVCacheBlocks 实例
        """
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """将 KVCacheBlocks 实例转换为 block_ids。

        Args:
            allow_none: 是否允许在没有块时返回 None

        Returns:
            tuple[list[int], ...]: 一个元组，其中：
                - 外层元组对应 KV 缓存组
                - 每个内层列表包含该组中块的 block_ids
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """从 KVCacheBlocks 实例中获取未哈希块的 block_ids。

        Returns:
            未哈希块的 block_id 列表
        """
        assert len(self.blocks) == 1, "只支持一个组"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """从 KVCacheBlocks 实例中获取所有组中未哈希块的 block_ids。

        Returns:
            每个组未哈希块的 block_id 列表
        """
        # 跳过填充块
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """创建一个新的空 KVCacheBlocks 实例。

        Returns:
            空 KVCacheBlocks 实例
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    """KV 缓存管理器。

    负责管理请求的 KV 缓存块分配、释放和缓存。
    通过协调器（coordinator）和块池（block_pool）实现
    复杂的缓存管理逻辑。

    Attributes:
        max_model_len: 最大模型长度
        enable_caching: 是否启用缓存
        use_eagle: 是否使用 EAGLE 推测解码
        log_stats: 是否记录统计信息
        metrics_collector: 指标收集器
        prefix_cache_stats: 前缀缓存统计信息
        coordinator: KV 缓存协调器
        num_kv_cache_groups: KV 缓存组数量
        block_pool: 块池
        kv_cache_config: KV 缓存配置
        empty_kv_cache_blocks: 预构造的空块实例
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        """初始化 KV 缓存管理器。

        Args:
            kv_cache_config: KV 缓存配置
            max_model_len: 最大模型长度
            hash_block_size: 计算块哈希的块大小
            enable_caching: 是否启用前缀缓存
            use_eagle: 是否使用 EAGLE
            log_stats: 是否记录统计信息
            enable_kv_cache_events: 是否启用 KV 缓存事件
            dcp_world_size: 解码上下文并行世界大小
            pcp_world_size: 预填充上下文并行世界大小
            metrics_collector: 可选的 KV 缓存指标收集器
        """
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: 使前缀缓存统计信息以 log_stats 为条件。当启用日志统计时，
        # 我们仍然保留此注释，因为未来可能会暴露一些配置。
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # 预构造的无块 KVCacheBlocks，调用者应通过 create_kv_cache_blocks
        # 使用这个而不是创建新的以避免 GC 开销。
        #
        # 我们使用嵌套元组来确保 empty_kv_cache_blocks 是不可变的。
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    @property
    def usage(self) -> float:
        """获取 KV 缓存使用率。

        Returns:
            KV 缓存使用率（0.0 到 1.0 之间）
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """获取（并重置）前缀缓存统计信息。

        Returns:
            当前前缀缓存统计信息，如果日志禁用则返回 None
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """获取请求的已计算（缓存）块。
        注意：已计算的块必须是完整的块。

        Args:
            request: 要获取已计算块的请求

        Returns:
            包含以下内容的元组：
                - 请求的已计算块列表
                - 已计算 token 的数量
        """
        # 当前缀缓存禁用或请求被标记为跳过 kv 缓存读取时，
        # 我们跳过查找前缀缓存命中（当请求需要 prompt logprobs
        # 或调用 pooling 模型进行所有 pooling 时会发生这种情况）
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # 注意：当所有 token 都命中缓存时，我们必须重新计算最后一个 token
        # 以获得 logits。因此，设置 max_cache_hit_length 为 prompt_length - 1。
        # 这会触发整个块的重新计算，而不仅仅是单个最后一个 token，因为
        # allocate_slots() 要求 num_computed_tokens 是 block-size 对齐的。
        # 移除这个限制可以在未来稍微提高性能。
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """为请求分配 slot 用于附加新 token。

        Args:
            request: 要分配 slot 的请求
            num_new_tokens: 要分配和计算的新 token 数量
            num_new_computed_tokens: 刚命中前缀缓存的新已计算 token 数量，
                                    不包括外部 token
            new_computed_blocks: 上述新已计算 token 的缓存块，
                               按 kv 缓存组分组为元组
            num_lookahead_tokens: 要分配的前瞻 token 数量。
                                这用于带有 kv-cache 的 spec decode proposers，
                                如 eagle
            num_external_computed_tokens: token 数量，其 KV 缓存不是由 vLLM
                                        缓存而是由 connector 缓存
            delay_cache_blocks: 是否跳过缓存块。这用于 P/D 场景，当分配
                              在 future step 中完成的 KV 传输使用的块时
            num_encoder_tokens: 为编码器 - 解码器模型（如 Whisper）的
                              交叉注意力分配的编码器 token 数量。
                              对于仅解码器模型，这应该是 0

        块布局：
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | 来自 vLLM 或 connector 的前缀缓存 token。|
        | 如果在滑动窗口外，可以安全地移除。        |
        ----------------------------------------------------------------------
        |   < 由 vLLM 缓存 >    | 不由          |
                                  | vLLM 缓存，但   |
        | ref_cnt  | ref_cnt not  | 由 connector |
        | increased| increased yet| 缓存          |
        ----------------------------------------------------------------------
        ```

        缩写：

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, 由 connector 缓存
        new       = num_new_tokens, 包括未验证的 draft token
        lookahead = num_lookahead_tokens
        ```

        注意：对于包括已验证和未验证 draft token 的新 token，
        我们只缓存已验证的 token（通过限制在 `request.num_tokens`）。

        分配有三个阶段：
        - 释放 `comp` 中不必要的块，检查是否有足够的空闲块
          （如果不足则返回 None）
        - 处理前缀 token（`comp + new_comp + ext_comp`）：
            - 释放不必要的块（例如滑动窗口外的块）
            - 为滑动窗口内的 `ext_comp` token 分配新块
        - 为要计算的 token（`new + lookahead`）分配新块

        Returns:
            新分配的块列表，如果分配失败则返回 None
        """
        # 当异步加载 KV 数据时，我们可能没有新 token 要计算，
        # 但仍需要为外部计算的 token 分配 slot
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "当没有外部计算 token 时，num_new_tokens 必须大于 0"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # 已计算 token 的数量是已计算 token 数量加上新前缀缓存命中数
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens,
            self.max_model_len,
        )

        # 释放在注意力计算期间跳过的块
        # （例如，滑动窗口外的 token）
        # 即使由于空闲块不足而无法调度此请求，我们也可以这样做。
        # 应在分配新块之前调用此函数以减少被驱逐的块数量。
        self.coordinator.remove_skipped_blocks(
            request.request_id, total_computed_tokens
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # 无法分配新块
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # 将新计算的块附加到请求块中，以避免新块无法分配的情况。
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D：如果需要从远程接收则延迟缓存块。
        # 为本地缓存的块更新状态。
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # 注意 (woosuk): 我们想要提交（缓存）最多
        # num_local_computed_tokens + num_external_computed_tokens + num_new_tokens，
        # 但必须排除"不可提交"的 token（例如可能被拒绝的 draft token）。
        # 因此，我们将数量限制在 `request.num_tokens`，确保只缓存"最终确定"的 token。
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """释放请求分配的块。
        我们按相反顺序释放块，以便在启用缓存时尾部块先被驱逐。

        Args:
            request: 要释放块的请求
        """
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """从 `blocks` 中移除不再需要的块并用 null_block 替换。

        Args:
            request_id: 请求 ID
            total_computed_tokens: 已计算 token 总数，包括
                                本地已计算 token 和外部已计算 token
        """
        self.coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """通过块 ID 从前缀缓存中驱逐块。

        Args:
            block_ids: 要从缓存中驱逐的块 ID 集合
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """重置前缀缓存。此函数可用于 RLHF 流程中
        在权重更新后使前缀缓存失效，或用于基准测试时
        重置前缀缓存状态。

        Returns:
            如果前缀缓存成功重置则返回 True，否则返回 False
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """计算每个 kv 缓存组的公共前缀块数量。

        该函数选择一个运行中的请求并遍历其块。
        如果所有分配了 KV 缓存的请求都共享一个块
        （即 ref_cnt 等于 req_to_blocks 中的条目数），
        则该块被认为是公共前缀块。

        注意 (woosuk): 分配了 KV 缓存的请求数量**大于或等于**
        当前 step 中调度的请求数量。这是因为分配了 KV 缓存只表示：
        1. 请求尚未完成，以及
        2. 请求持有其块未释放。

        虽然所有已调度的请求必须分配了 KV 缓存，但反之不一定成立。
        可能存在分配了 KV 缓存但在当前 step 中未调度的请求。

        这可能导致一种边缘情况，即公共前缀块的数量为 0，
        即使所有已调度的请求共享一个公共前缀。这是因为可能
        存在不共享公共前缀的未调度请求。目前，这种情况无法轻易检测，
        所以函数在这种情况下返回 0。

        Args:
            running_request_id: 任何运行中请求的请求 ID，
                              用于识别公共前缀块

        Returns:
            list[int]: 每个 kv 缓存组的公共前缀块数量
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """从块池获取 KV 缓存事件。

        Returns:
            KV 缓存事件列表
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """获取请求的块。

        Args:
            request_id: 请求 ID

        Returns:
            请求的 KVCacheBlocks 实例
        """
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """获取请求的 block ids。

        Args:
            request_id: 请求 ID

        Returns:
            请求的 block_ids 元组
        """
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """为请求缓存块（如果启用）。

        Args:
            request: 要缓存块的请求
            num_computed_tokens: 已计算 token 数量，包括
                               已缓存的 token 和要缓存的 token
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        """创建 KVCacheBlocks 实例。

        只为非空块创建新的 KVCacheBlocks。

        Args:
            blocks: 块元组

        Returns:
            KVCacheBlocks 实例或预构造的空实例
        """
        # 只为非空块创建新的 KVCacheBlocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def take_new_block_ids(self) -> list[int]:
        """取出并返回需要清零的新注意力块 ID。

        Returns:
            新块 ID 列表
        """
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    def new_step_starts(self) -> None:
        """在新 step 开始时调用。"""
        self.coordinator.new_step_starts()
