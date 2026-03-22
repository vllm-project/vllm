# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""指标统计数据模块。

本模块定义了 vLLM 引擎使用的各种统计数据类，负责：
- 定义缓存命中统计基类和数据类
- 提供前缀缓存和多模态缓存指标
- 定义调度器统计数据
- 跟踪请求状态和完成统计
- 支持 LoRA 适配器状态管理
- 提供迭代统计和性能指标

主要类：
- BaseCacheStats: 缓存命中统计基类
- CachingMetrics: 缓存命中率指标
- PrefixCacheStats: 前缀缓存统计
- MultiModalCacheStats: 多模态缓存统计
- KVCacheEvictionEvent: KV 缓存驱逐事件
- SchedulerStats: 调度器统计
- RequestStateStats: 请求状态统计
- FinishedRequestStats: 完成请求统计
- PromptTokenStats: 提示 token 统计
- IterationStats: 迭代统计
- LoRAStats: LoRA 状态
- LoRARequestStates: LoRA 请求状态管理
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreEvent, EngineCoreOutput, FinishReason


@dataclass
class BaseCacheStats:
    """缓存命中统计基类。

    存储缓存命中统计数据。

    Attributes:
        reset: 缓存是否被重置
        requests: 本次更新的请求数量
        queries: 本次更新的查询数量
        hits: 本次更新的命中数量
    """

    reset: bool = False
    """缓存是否被重置。"""

    requests: int = 0
    """本次更新中的请求数量。"""

    queries: int = 0
    """本次更新中的查询数量。"""

    hits: int = 0
    """本次更新中的命中数量。"""


class CachingMetrics:
    """最近 N 个请求的缓存命中率指标。

    基于滑动窗口机制计算缓存命中率，
    当请求数量超过最大值时，移除最旧的请求。

    Attributes:
        max_recent_requests: 最大最近请求数量，默认为 1000
        aggregated_requests: 聚合的请求数量
        aggregated_query_total: 聚合的查询总数
        aggregated_query_hit: 聚合的命中数
        query_queue: 最近请求的队列

    Args:
        max_recent_requests: 聚合的最近请求数量，默认为 1000
    """

    def __init__(self, max_recent_requests: int = 1000) -> None:
        """初始化缓存指标。

        Args:
            max_recent_requests: 最大最近请求数量
        """
        super().__init__()

        self.max_recent_requests = max_recent_requests
        # 当前聚合的值
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0

        # 最近请求的 (requests, queries, hits) 队列
        self.query_queue = deque[tuple[int, int, int]]()

    def observe(self, stats: BaseCacheStats):
        """观察前缀缓存统计。

        当新请求被调度并查找已计算的块时调用此函数。

        当请求数量超过 `max_recent_requests` 时，
        从指标中移除最旧的请求集。

        Args:
            stats: 前缀缓存统计
        """
        # 在当前更新之前调用了 reset_prefix_cache
        # 在聚合当前统计之前重置指标
        if stats.reset:
            self.reset()

        # 避免追加空统计导致有用信息被滑动窗口踢出
        if stats.requests == 0:
            return

        # 更新指标
        self.query_queue.append((stats.requests, stats.queries, stats.hits))
        self.aggregated_requests += stats.requests
        self.aggregated_query_total += stats.queries
        self.aggregated_query_hit += stats.hits

        # 移除最旧的统计直到请求数量不超过限制
        # 注意：我们保留最新添加的统计
        while (
            len(self.query_queue) > 1
            and self.aggregated_requests > self.max_recent_requests
        ):
            old_requests, old_queries, old_hits = self.query_queue.popleft()
            self.aggregated_requests -= old_requests
            self.aggregated_query_total -= old_queries
            self.aggregated_query_hit -= old_hits

    def reset(self):
        """重置指标。"""
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        self.query_queue.clear()

    @property
    def empty(self) -> bool:
        """返回是否未观察到任何请求。

        Returns:
            如果未观察到请求返回 True
        """
        return self.aggregated_requests == 0

    @property
    def hit_rate(self) -> float:
        """计算过去 N 个请求的命中率。

        Returns:
            命中率 (0.0-1.0)
        """
        if self.aggregated_query_total == 0:
            return 0.0
        return self.aggregated_query_hit / self.aggregated_query_total


@dataclass
class PrefixCacheStats(BaseCacheStats):
    """存储前缀缓存命中统计。

    - `reset`: 是否调用了 `reset_prefix_cache`
    - `queries`: 指被查询的 token 数量

    Attributes:
        preempted_requests: 本次更新中被抢占的请求数量
        preempted_queries: 被抢占请求的查询数量
        preempted_hits: 被抢占请求的命中数量
    """

    preempted_requests: int = 0
    """本次更新中被抢占的请求数量。"""

    preempted_queries: int = 0
    """被抢占请求的 `queries` 数量。"""

    preempted_hits: int = 0
    """被抢占请求的 `hits` 数量。"""

    def record(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        """将请求信息聚合到统计中。

        Args:
            num_tokens: token 数量
            num_hits: 命中数量
            preempted: 是否是抢占的请求
        """
        if preempted:
            # 之前被抢占的请求
            self.preempted_requests += 1
            self.preempted_queries += num_tokens
            self.preempted_hits += num_hits
        else:
            # 新请求
            self.requests += 1
            self.queries += num_tokens
            self.hits += num_hits


@dataclass
class MultiModalCacheStats(BaseCacheStats):
    """存储多模态缓存命中统计。

    - `reset`: 是否调用了 `reset_mm_cache`
    - `queries`: 指被查询的多模态数据项数量

    Attributes:
        requests: 请求数量
        queries: 查询数量
        hits: 命中数量
    """

    def record(self, num_queries: int, num_hits: int) -> None:
        """将请求信息聚合到统计中。

        Args:
            num_queries: 查询数量
            num_hits: 命中数量
        """
        self.requests += 1
        self.queries += num_queries
        self.hits += num_hits


@dataclass
class KVCacheEvictionEvent:
    """单个 KV 缓存块驱逐采样。

    Attributes:
        lifetime_seconds: 块的生命周期（秒）
        idle_seconds: 空闲时间（秒）
        reuse_gaps_seconds: 重用间隔元组（秒）
    """

    lifetime_seconds: float
    idle_seconds: float
    reuse_gaps_seconds: tuple[float, ...]


@dataclass
class SchedulerStats:
    """与调度器相关的统计。

    Attributes:
        num_running_reqs: 运行中的请求数量
        num_waiting_reqs: 等待中的请求数量
        step_counter: 步骤计数器（用于内部 DP 负载均衡）
        current_wave: 当前波次（用于内部 DP 负载均衡）
        kv_cache_usage: KV 缓存使用率
        encoder_cache_usage: 编码器缓存使用率
        prefix_cache_stats: 前缀缓存统计
        connector_prefix_cache_stats: 连接器前缀缓存统计
        kv_cache_eviction_events: KV 缓存驱逐事件列表
        spec_decoding_stats: 推测解码统计
        kv_connector_stats: KV 连接器统计
        waiting_lora_adapters: 等待中的 LoRA 适配器
        running_lora_adapters: 运行中的 LoRA 适配器
        cudagraph_stats: CUDA Graph 统计
        perf_stats: 性能统计
    """

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    # 这些用于内部 DP 负载均衡
    step_counter: int = 0
    current_wave: int = 0

    kv_cache_usage: float = 0.0
    encoder_cache_usage: float = 0.0

    prefix_cache_stats: PrefixCacheStats = field(default_factory=PrefixCacheStats)
    connector_prefix_cache_stats: PrefixCacheStats | None = None

    kv_cache_eviction_events: list[KVCacheEvictionEvent] = field(default_factory=list)

    spec_decoding_stats: SpecDecodingStats | None = None
    kv_connector_stats: dict[str, Any] | None = None

    waiting_lora_adapters: dict[str, int] = field(default_factory=dict)
    running_lora_adapters: dict[str, int] = field(default_factory=dict)

    cudagraph_stats: CUDAGraphStat | None = None

    perf_stats: PerfStats | None = None


@dataclass
class RequestStateStats:
    """需要在增量更新中跟踪的统计。

    Attributes:
        num_generation_tokens: 生成的 token 数量
        arrival_time: 到达时间（引擎前端时间戳，挂钟时间）
        queued_ts: 入队时间戳（引擎核心时间戳，单调时间）
        scheduled_ts: 调度时间戳
        first_token_ts: 首个 token 时间戳
        last_token_ts: 最后一个 token 时间戳
        first_token_latency: 首个 token 延迟
        is_corrupted: 请求是否损坏（logits 中出现 NaN）
    """

    num_generation_tokens: int = 0

    # 这是引擎前端时间戳（挂钟时间）
    arrival_time: float = 0.0

    # 这些是引擎核心时间戳（单调时间）
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0

    # 首个 token 延迟
    first_token_latency: float = 0.0

    # 跟踪此请求是否损坏（logits 中出现 NaN）
    is_corrupted: bool = False


@dataclass
class FinishedRequestStats:
    """与已完成请求相关的统计。

    Attributes:
        finish_reason: 完成原因
        e2e_latency: 端到端延迟
        num_prompt_tokens: 提示 token 数量
        num_generation_tokens: 生成 token 数量
        max_tokens_param: 最大 token 参数
        queued_time: 排队时间
        prefill_time: 预填充时间
        inference_time: 推理时间
        decode_time: 解码时间
        mean_time_per_output_token: 每个输出 token 的平均时间
        is_corrupted: 是否损坏
        num_cached_tokens: 缓存的 token 数量
    """

    finish_reason: "FinishReason"
    e2e_latency: float = 0.0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    max_tokens_param: int | None = None
    queued_time: float = 0.0
    prefill_time: float = 0.0
    inference_time: float = 0.0
    decode_time: float = 0.0
    mean_time_per_output_token: float = 0.0
    is_corrupted: bool = False
    num_cached_tokens: int = 0


@dataclass
class PromptTokenStats:
    """按来源分解的提示 token 统计。

    Fields:
        computed: 本地预填充的 token（实际计算工作）
        local_cache_hit: 来自本地前缀缓存的 token
        external_kv_transfer: 来自外部 KV 传输的 token
        cached_tokens: 预填充期间跳过的 token（来自调度器）
        recomputed_tokens: 被重新计算的缓存 token（见下文说明）
        total: 总提示 token 数量

    Invariants:
        computed + local_cache_hit + external_kv_transfer - recomputed_tokens = total
        local_cache_hit + external_kv_transfer - recomputed_tokens = cached_tokens

    注意：当所有 token 都被缓存时，调度器会将 num_cached_tokens 减 1，
    强制模型重新计算最后一个 token，因为模型至少需要一个输入 token
    才能运行前向传播。
    """

    ALL_SOURCES: tuple[str, ...] = (
        "local_compute",
        "local_cache_hit",
        "external_kv_transfer",
    )

    computed: int = 0
    local_cache_hit: int = 0
    external_kv_transfer: int = 0
    cached_tokens: int = 0
    recomputed_tokens: int = 0
    total: int = 0

    def update_from_output(
        self,
        num_cached_tokens: int,
        num_external_computed_tokens: int,
        prompt_len: int,
    ) -> None:
        """根据预填充输出更新统计。

        Args:
            num_cached_tokens: 缓存的 token 数量
            num_external_computed_tokens: 外部计算的 token 数量
            prompt_len: 提示长度
        """
        # 当所有 token 都被缓存时，调度器将 num_cached_tokens 减 1
        # 强制模型重新计算最后一个 token，因为模型至少需要一个输入 token
        # 才能运行前向传播
        recomputed = 1 if (num_cached_tokens + 1 == prompt_len) else 0

        self.computed += prompt_len - num_cached_tokens
        self.external_kv_transfer += num_external_computed_tokens
        self.local_cache_hit += (
            num_cached_tokens + recomputed - num_external_computed_tokens
        )
        self.cached_tokens += num_cached_tokens
        self.recomputed_tokens += recomputed
        self.total += prompt_len

    def get_by_source(self, source: str) -> int:
        """按来源标签获取 token 数量。

        Args:
            source: 来源标签

        Returns:
            该来源的 token 数量

        Raises:
            ValueError: 如果来源未知
        """
        source_map = {
            "local_compute": self.computed,
            "local_cache_hit": self.local_cache_hit,
            "external_kv_transfer": self.external_kv_transfer,
        }
        if source not in source_map:
            raise ValueError(f"未知来源：{source}")
        return source_map[source]


class IterationStats:
    """与单次 EngineCoreOutputs 相关的统计。

    Attributes:
        iteration_timestamp: 迭代时间戳
        num_generation_tokens: 生成 token 数量
        prompt_token_stats: 提示 token 统计
        num_preempted_reqs: 被抢占的请求数量
        finished_requests: 已完成请求列表
        max_num_generation_tokens_iter: 每次迭代的最大生成 token 数列表
        n_params_iter: 每次迭代的 n 参数列表
        time_to_first_tokens_iter: 每次迭代的首 token 时间列表
        inter_token_latencies_iter: 每次迭代的 token 间延迟列表
        num_corrupted_reqs: 损坏的请求数量
    """

    def __init__(self):
        """初始化迭代统计。"""
        self.iteration_timestamp = time.time()
        self.num_generation_tokens = 0
        self.prompt_token_stats = PromptTokenStats()
        self.num_preempted_reqs = 0
        self.finished_requests: list[FinishedRequestStats] = []
        self.max_num_generation_tokens_iter: list[int] = []
        self.n_params_iter: list[int] = []
        self.time_to_first_tokens_iter: list[float] = []
        self.inter_token_latencies_iter: list[float] = []
        self.num_corrupted_reqs: int = 0

    def __repr__(self) -> str:
        """返回字符串表示。

        Returns:
            包含所有属性的字符串
        """
        field_to_value_str = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({field_to_value_str})"

    @property
    def num_prompt_tokens(self) -> int:
        """总提示 token 数量（向后兼容）。

        Returns:
            总提示 token 数量
        """
        return self.prompt_token_stats.total

    def _time_since(self, start: float) -> float:
        """计算相对于此迭代时间戳的时间间隔。

        Args:
            start: 开始时间

        Returns:
            时间间隔（秒）
        """
        return self.iteration_timestamp - start

    def update_from_output(
        self,
        output: "EngineCoreOutput",
        engine_core_timestamp: float,
        is_prefilling: bool,
        prompt_len: int,
        req_stats: RequestStateStats,
        lora_states: "LoRARequestStates",
        lora_name: str | None,
    ):
        """根据输出更新统计。

        Args:
            output: 引擎核心输出
            engine_core_timestamp: 引擎核心时间戳
            is_prefilling: 是否处于预填充阶段
            prompt_len: 提示长度
            req_stats: 请求状态统计
            lora_states: LoRA 请求状态
            lora_name: LoRA 名称
        """
        num_new_generation_tokens = len(output.new_token_ids)

        self.num_generation_tokens += num_new_generation_tokens
        if is_prefilling:
            self.prompt_token_stats.update_from_output(
                num_cached_tokens=output.num_cached_tokens,
                num_external_computed_tokens=output.num_external_computed_tokens,
                prompt_len=prompt_len,
            )

            first_token_latency = self._time_since(req_stats.arrival_time)
            self.time_to_first_tokens_iter.append(first_token_latency)
            req_stats.first_token_latency = first_token_latency

        req_stats.num_generation_tokens += num_new_generation_tokens

        # 跟踪此请求是否损坏（每个请求仅检查一次）
        # 如果已标记为损坏则提前退出，避免重复检查
        if (
            envs.VLLM_COMPUTE_NANS_IN_LOGITS
            and not req_stats.is_corrupted
            and output.num_nans_in_logits > 0
        ):
            req_stats.is_corrupted = True

        # 处理请求级别的引擎核心事件
        if output.events is not None:
            self.update_from_events(
                output.request_id,
                output.events,
                is_prefilling,
                req_stats,
                lora_states,
                lora_name,
            )

        # 处理批次级别的"新 token"引擎核心事件
        if is_prefilling:
            req_stats.first_token_ts = engine_core_timestamp
        else:
            itl = engine_core_timestamp - req_stats.last_token_ts
            self.inter_token_latencies_iter.append(itl)

        req_stats.last_token_ts = engine_core_timestamp

    def update_from_events(
        self,
        req_id: str,
        events: list["EngineCoreEvent"],
        is_prefilling: bool,
        req_stats: RequestStateStats,
        lora_states: "LoRARequestStates",
        lora_name: str | None,
    ):
        """根据事件更新统计。

        Args:
            req_id: 请求 ID
            events: 引擎核心事件列表
            is_prefilling: 是否处于预填充阶段
            req_stats: 请求状态统计
            lora_states: LoRA 请求状态
            lora_name: LoRA 名称
        """
        # 避免循环依赖
        from vllm.v1.engine import EngineCoreEventType

        for event in events:
            if event.type == EngineCoreEventType.QUEUED:
                req_stats.queued_ts = event.timestamp
                lora_states.request_waiting(req_id, lora_name)
            elif event.type == EngineCoreEventType.SCHEDULED:
                if req_stats.scheduled_ts == 0.0:  # 忽略抢占
                    req_stats.scheduled_ts = event.timestamp
                lora_states.request_running(req_id, lora_name)
            elif event.type == EngineCoreEventType.PREEMPTED:
                self.num_preempted_reqs += 1
                lora_states.request_waiting(req_id, lora_name)

    def update_from_finished_request(
        self,
        finish_reason: "FinishReason",
        num_prompt_tokens: int,
        max_tokens_param: int | None,
        req_stats: RequestStateStats,
        num_cached_tokens: int = 0,
    ):
        """从已完成的请求更新统计。

        Args:
            finish_reason: 完成原因
            num_prompt_tokens: 提示 token 数量
            max_tokens_param: 最大 token 参数
            req_stats: 请求状态统计
            num_cached_tokens: 缓存的 token 数量
        """
        e2e_latency = self._time_since(req_stats.arrival_time)

        # 排队区间是从第一个 QUEUED 事件到第一个 SCHEDULED
        queued_time = req_stats.scheduled_ts - req_stats.queued_ts

        # 预填充区间是从第一个 SCHEDULED 到第一个 NEW_TOKEN
        # 预填充期间的任何抢占都包含在区间内
        prefill_time = req_stats.first_token_ts - req_stats.scheduled_ts

        # 解码区间是从第一个 NEW_TOKEN 到最后一个 NEW_TOKEN
        # 解码期间的任何抢占都包含在内
        decode_time = req_stats.last_token_ts - req_stats.first_token_ts

        # 推理区间是从第一个 SCHEDULED 到最后一个 NEW_TOKEN
        # 预填充或解码期间的任何抢占都包含在内
        inference_time = req_stats.last_token_ts - req_stats.scheduled_ts

        # 不计入预填充阶段生成的 token
        mean_time_per_output_token = (
            decode_time / (req_stats.num_generation_tokens - 1)
            if req_stats.num_generation_tokens - 1 > 0
            else 0
        )

        finished_req = FinishedRequestStats(
            finish_reason=finish_reason,
            e2e_latency=e2e_latency,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=req_stats.num_generation_tokens,
            max_tokens_param=max_tokens_param,
            queued_time=queued_time,
            prefill_time=prefill_time,
            inference_time=inference_time,
            decode_time=decode_time,
            mean_time_per_output_token=mean_time_per_output_token,
            is_corrupted=req_stats.is_corrupted,
            num_cached_tokens=num_cached_tokens,
        )
        self.finished_requests.append(finished_req)

        # 在请求完成时统计损坏的请求（每个请求仅一次）
        if req_stats.is_corrupted:
            self.num_corrupted_reqs += 1


class LoRAStats:
    """跟踪单个 LoRA 的等待和运行请求 ID。

    Attributes:
        waiting: 等待中的请求 ID 集合
        running: 运行中的请求 ID 集合
    """

    def __init__(self):
        """初始化 LoRA 统计。"""
        self.waiting: set[str] = set()
        self.running: set[str] = set()

    def update(self, req_id: str, waiting: bool, running: bool):
        """更新 LoRA 状态。

        Args:
            req_id: 请求 ID
            waiting: 是否等待中
            running: 是否运行中
        """
        assert not (waiting and running)
        if waiting:
            self.waiting.add(req_id)
        else:
            self.waiting.discard(req_id)

        if running:
            self.running.add(req_id)
        else:
            self.running.discard(req_id)

    @property
    def empty(self) -> bool:
        """返回是否没有任何请求。

        Returns:
            如果没有请求返回 True
        """
        return not (self.waiting or self.running)


class LoRARequestStates:
    """每个 LoRA 的运行和等待请求计数。

    Attributes:
        log_stats: 是否记录统计
        requests: 每个 LoRA 名称的统计字典
    """

    def __init__(self, log_stats: bool = False):
        """初始化 LoRA 请求状态。

        Args:
            log_stats: 是否记录统计
        """
        self.log_stats = log_stats
        self.requests: defaultdict[str, LoRAStats] = defaultdict(LoRAStats)

    def _request_update(
        self, req_id: str, lora_name: str | None, waiting: bool, running: bool
    ):
        """更新请求状态。

        Args:
            req_id: 请求 ID
            lora_name: LoRA 名称
            waiting: 是否等待中
            running: 是否运行中
        """
        if not self.log_stats or lora_name is None:
            return

        lora_stats = self.requests[lora_name]
        lora_stats.update(req_id, waiting, running)
        if lora_stats.empty:
            del self.requests[lora_name]

    def request_waiting(self, req_id: str, lora_name: str | None):
        """标记请求为等待状态。

        Args:
            req_id: 请求 ID
            lora_name: LoRA 名称
        """
        self._request_update(req_id, lora_name, waiting=True, running=False)

    def request_running(self, req_id: str, lora_name: str | None):
        """标记请求为运行状态。

        Args:
            req_id: 请求 ID
            lora_name: LoRA 名称
        """
        self._request_update(req_id, lora_name, waiting=False, running=True)

    def request_finished(self, req_id: str, lora_name: str | None):
        """标记请求为完成状态。

        Args:
            req_id: 请求 ID
            lora_name: LoRA 名称
        """
        self._request_update(req_id, lora_name, waiting=False, running=False)

    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        """更新调度器统计。

        Args:
            scheduler_stats: 调度器统计
        """
        if not self.log_stats or scheduler_stats is None:
            return
        for lora_name, stats in self.requests.items():
            scheduler_stats.waiting_lora_adapters[lora_name] = len(stats.waiting)
            scheduler_stats.running_lora_adapters[lora_name] = len(stats.running)
