# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""编码器缓存管理器模块。

本模块实现了多模态模型的编码器输出缓存管理功能，负责：
- 管理多模态编码器输出的缓存生命周期（如视觉嵌入）
- 提供内存感知的缓存管理，避免重复计算编码器输出
- 支持细粒度的内存管理和多模态输入的块处理
- 通过 LRU 策略管理缓存驱逐

主要类：
- EncoderCacheManager: 编码器缓存管理器
- EncoderDecoderCacheManager: 编码器 - 解码器缓存管理器

应用场景：
- 视觉 - 语言模型（如 LLaVA）缓存图像编码器输出
- 任何多模态模型中编码器计算开销大且可缓存的场景
"""

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig

logger = init_logger(__name__)


class EncoderCacheManager:
    """编码器缓存管理器。

    管理多模态模型（如视觉 - 语言模型）的编码器输出缓存。
    处理请求处理过程中多模态编码器输出（如视觉嵌入）的生命周期。
    提供内存感知的缓存，当相同的多模态输入出现在请求处理的不同阶段时，
    避免重新计算编码器输出。

    缓存以请求中各个多模态输入项的粒度操作，允许细粒度的内存管理。
    缓存启用时，在不同请求之间共享相同多模态数据项（通过哈希值识别）的嵌入，
    当没有新嵌入空间时在分配时进行驱逐。具有零引用的最旧缓存嵌入将首先被驱逐。

    注意：
        EncoderCacheManager 操作在多模态嵌入级别而不是编码器 token 级别。
        这意味着嵌入之间的所有中断/文本 token 不考虑缓存大小和空闲槽位数量。

    Attributes:
        cache_size: 以编码器嵌入为单位的总缓存容量
        num_free_slots: 当前可用缓存槽位数量（编码器嵌入）
        num_freeable_slots: 可通过驱逐零引用条目立即回收的容量
        cached: 从 mm_hash 到引用该缓存条目的请求 ID 集合的映射
        freeable: 可释放条目列表（mm_hash, num_encoder_embeds 元组）
        freed: 自上次调用 get_freed_mm_hashes() 以来实际被驱逐的 mm_hash 列表

    Args:
        cache_size: 限制缓存大小，以输入序列中的编码器嵌入数量为单位
    """

    def __init__(self, cache_size: int):
        """初始化编码器缓存管理器。

        Args:
            cache_size: 缓存容量（编码器嵌入数量）
        """
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size

        # mm_hash -> 引用该 mm_data 的请求 ID 集合
        self.cached: dict[str, set[str]] = {}

        # mm_hash -> mm_data 的 num_encoder_embeds
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

    def reset(self) -> None:
        """重置编码器缓存到初始状态。

        清除所有缓存的编码器输出并重置容量跟踪。
        在模型权重更新时调用以使过时的嵌入失效。
        """
        self.cached.clear()
        self.freeable.clear()
        self.freed.clear()
        self.num_free_slots = self.cache_size
        self.num_freeable_slots = self.cache_size

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """检查特定多模态输入的编码器输出是否已缓存。

        如果编码器输出已缓存，更新 cached 以将请求 ID 添加到引用该缓存
        编码器输出的请求 ID 集合中。
        如果编码器输出之前未被任何请求引用，更新 freeable 和 num_freeable_slots。

        Args:
            request: 包含多模态输入的请求
            input_id: 请求中多模态输入的索引

        Returns:
            如果此输入的编码器输出已缓存则返回 True
        """
        mm_hash = request.mm_features[input_id].identifier
        # 完全不在缓存中
        if mm_hash not in self.cached:
            return False

        # 已缓存但目前未被任何请求引用
        if not self.cached[mm_hash]:
            num_encoder_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_encoder_embeds

        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        """检查是否有足够的缓存空间用于多模态输入。

        如果有足够的空间，返回 True 并更新 EncoderCacheManager 状态。
        如果 num_free_slots 中没有足够的空闲空间，但 num_freeable_slots 中有
        足够的可回收空间，则从 freeable 中驱逐条目（将 mm_hash 追加到 freed），
        直到有足够的可用空间，然后返回 True。较旧的条目首先被驱逐。
        仅当请求的 token 数量超过空闲和可回收容量总和时才返回 False。

        Args:
            request: 包含多模态输入的请求
            input_id: 请求中多模态输入的索引
            encoder_compute_budget: 调用此方法时允许计算的编码器嵌入数量
            num_embeds_to_schedule: 调用此方法时已计划分配缓存空间的编码器嵌入数量

        Returns:
            如果有足够的容量容纳此输入的编码器输出（可能在回收 freeable 条目后）
            则返回 True；否则返回 False

        注意：
            此方法不分配编码器输出的物理内存，仅更新 EncoderCacheManager 状态。
        """
        num_embeds = request.get_num_encoder_embeds(input_id)

        # 计算预算不足
        if num_embeds > encoder_compute_budget:
            return False

        num_embeds += num_embeds_to_schedule

        # 有足够的空闲槽位
        if num_embeds <= self.num_free_slots:
            return True

        # 没有足够的可回收槽位
        if num_embeds > self.num_freeable_slots:
            return False

        # 没有足够的空闲槽位但有足够的可回收槽位
        # 注意：驱逐在这里发生，但物理内存要等到调度器通知模型运行器时才释放
        while num_embeds > self.num_free_slots:
            mm_hash, num_free_embeds = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_embeds
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        """为多模态输入的编码器输出分配缓存空间。

        这预留了缓存空间用于存储指定多模态输入的编码器输出。
        实际的编码器输出存储发生在模型运行器中；此方法更新管理器的记账。

        注意：
            此方法假设 can_allocate() 对相同输入返回了 True。
        """
        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_embeds = request.get_num_encoder_embeds(input_id)

        # 注意：编码器缓存应该始终有足够的空间用于已调度的编码器输入，
        # 因为驱逐发生在 can_allocate() 时
        assert self.num_free_slots >= num_encoder_embeds
        assert self.num_freeable_slots >= num_encoder_embeds

        self.cached[mm_hash].add(request_id)
        self.num_free_slots -= num_encoder_embeds
        self.num_freeable_slots -= num_encoder_embeds

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """获取请求的所有缓存多模态输入 ID。

        返回其 mm_hash 存在于缓存映射中的输入 ID 集合。
        包括当前未引用的条目（因此在 freeable 中）；对于这些条目，
        为此请求的释放将是空操作。

        Returns:
            缓存的输入 ID 集合
        """
        return {
            input_id
            for input_id in range(len(request.mm_features))
            if request.mm_features[input_id].identifier in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """释放请求对编码器输入（mm_data）的引用。

        当相应 mm_hash 的引用集合变为空时，条目被追加到 freeable，
        num_freeable_slots 增加该输入的编码器嵌入数量。
        条目不会立即被物理驱逐，直到需要容量时（例如通过 can_allocate）。

        Args:
            request: 请求
            input_id: 多模态输入索引
        """
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # mm_hash 不在缓存中或 req_id 集合为空
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            self.freeable[mm_hash] = num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds

    def free(self, request: Request) -> None:
        """释放请求持有的所有编码器输入缓存引用。

        对于每个缓存的输入 ID，调用 free_encoder_input。
        数据保留在内存中，直到未来的 can_allocate 调用触发驱逐。

        通常在请求完成、取消或中止时调用。

        Args:
            request: 要释放的请求
        """
        input_ids = self.get_cached_input_ids(request)
        for input_id in input_ids:
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """获取并清除最近释放的编码器缓存条目列表。

        Returns:
            自上次调用以来实际被驱逐的 mm_hash 字符串列表，
            供调度器通知工作节点可以从其缓存中移除哪些编码器输出。
            内部列表在此调用后清除。
        """
        freed = self.freed
        self.freed = []
        return freed


def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    mm_max_toks_per_item: Mapping[str, int],
) -> tuple[int, int]:
    """为多模态模型计算编码器缓存预算。

    基于模型和调度器配置计算编码器缓存预算。

    Args:
        scheduler_config: 调度器配置
        mm_max_toks_per_item: 每种非文本模态每个项目的最大 token 数量

    Returns:
        - 编码器计算的 token 预算，以输入序列中的 token 数量为单位
        - 编码器缓存空间预算，以输入序列中的 token 数量为单位
    """
    if not mm_max_toks_per_item:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized."
        )
        return 0, 0

    max_tokens_per_mm_item = max(mm_max_toks_per_item.values())

    if (
        scheduler_config.disable_chunked_mm_input
        and max_tokens_per_mm_item > scheduler_config.max_num_batched_tokens
    ):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens."
        )

    encoder_compute_budget = max(
        scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
    )
    encoder_cache_size = max(
        scheduler_config.encoder_cache_size, max_tokens_per_mm_item
    )

    return encoder_compute_budget, encoder_cache_size


# NOTE (NickLucche): 编码器 - 解码器模型的临时实现，仅将管理器用于调度目的。
# 编码器 - 解码器模型最终将使用缓存，此类将合并到 EncoderCacheManager 中，
# 因为与多模态模型的差异正在缩小。
class EncoderDecoderCacheManager(EncoderCacheManager):
    """编码器 - 解码器缓存管理器。

    用于编码器 - 解码器模型的缓存管理，与多模态模型的缓存管理略有不同。
    主要差异在于缓存行为更加简单，不涉及复杂的引用计数。
    """

    def __init__(self, cache_size: int):
        """初始化编码器 - 解码器缓存管理器。

        Args:
            cache_size: 缓存容量
        """
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.allocated: list[str] = []
        self.to_free: list[str] = []

    def reset(self) -> None:
        """重置编码器缓存到初始状态。"""
        self.num_free_slots = self.cache_size
        self.allocated.clear()
        self.to_free.clear()

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """检查和更新缓存（编码器 - 解码器模型不使用缓存）。

        Returns:
            总是返回 False
        """
        return False

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        """检查是否可以分配缓存空间。

        Args:
            request: 请求
            input_id: 多模态输入索引
            encoder_compute_budget: 编码器计算预算
            num_embeds_to_schedule: 要调度的嵌入数量

        Returns:
            如果有足够的空闲槽位则返回 True
        """
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        # 计算预算不足
        if num_encoder_embeds > encoder_compute_budget:
            return False

        num_encoder_embeds += num_embeds_to_schedule
        # 有足够的空闲槽位
        return num_encoder_embeds <= self.num_free_slots

    def allocate(self, request: Request, input_id: int) -> None:
        """分配缓存空间。

        Args:
            request: 请求
            input_id: 多模态输入索引
        """
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots -= num_encoder_embeds

        mm_hash = request.mm_features[input_id].identifier
        self.allocated.append(mm_hash)

    def free(self, request: Request) -> None:
        """释放请求的所有编码器输入缓存。

        Args:
            request: 请求
        """
        for input_id in range(len(request.mm_features)):
            self.free_encoder_input(request, input_id)

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """获取缓存的输入 ID。

        Returns:
            所有输入 ID 的集合
        """
        return set(range(len(request.mm_features)))

    def get_freed_mm_hashes(self) -> list[str]:
        """获取并清除最近释放的 mm_hash 列表。

        由于编码器 - 解码器模型不使用编码器缓存，我们可以在此处释放条目。
        实际的释放发生在运行器中，在模型执行之前。
        因此，freeable 充当缓冲区，仅在模型执行后释放条目，
        模拟 EncoderCacheManager 的状态转换。

        Returns:
            要释放的 mm_hash 列表
        """
        to_free = self.to_free
        self.to_free = self.allocated
        self.allocated = []
        return to_free

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """释放编码器输入。

        Args:
            request: 请求
            input_id: 多模态输入索引
        """
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots += num_encoder_embeds
