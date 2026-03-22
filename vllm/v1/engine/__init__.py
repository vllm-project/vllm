# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM V1 引擎核心数据结构和类型定义模块。

本模块定义了 vLLM V1 引擎的核心数据结构，包括请求、输出、事件等。
这些结构使用 msgspec 进行高效序列化，用于引擎内部进程间通信。
"""

import enum
import time
from collections.abc import Mapping
from typing import Any, Literal

import msgspec
import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
from vllm.v1.serial_utils import UtilityResult

# 生成暂停模式的类型定义
# - "abort": 立即中止所有进行中的请求（默认）。
# - "wait": 等待进行中的请求完成后再暂停。
# - "keep": 冻结队列中的请求，它们在 resume_generation() 时恢复。
PauseMode = Literal["abort", "wait", "keep"]

# 请求完成原因的可能取值，构成外部 API 的一部分
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")

# 弹性 EP 通知的调用 ID 常量
EEP_NOTIFICATION_CALL_ID = -1


class EEPNotificationType(enum.Enum):
    """弹性 EP（Elastic Expert Parallel）通知类型枚举。

    用于在弹性扩展/缩容过程中协调不同引擎核心进程之间的状态。
    """
    # 新核心引擎初始化就绪
    NEW_CORE_ENGINES_INIT_READY = "NEW_CORE_ENGINES_INIT_READY"
    # 新核心引擎权重初始化就绪
    NEW_CORE_ENGINES_WEIGHTS_INIT_READY = "NEW_CORE_ENGINES_WEIGHTS_INIT_READY"
    # 重配置完成
    RECONFIGURE_FINISHED = "RECONFIGURE_FINISHED"
    # 关闭完成
    SHUTDOWN_COMPLETE = "SHUTDOWN_COMPLETE"


class FinishReason(enum.IntEnum):
    """
    请求完成的原因枚举 - 停止、长度、中止、错误或重复。

    使用 Int 而非 Str 以获得更紧凑的序列化。

    stop - 生成了停止字符串
    length - 达到 max_tokens 或 max_model_len
    abort - 被客户端中止
    error - 可重试的请求级内部错误（如 KV 负载失败）。
            不变式：始终转换为 500 Internal Server Error。
    repetition - 检测到重复的 token 模式（幻觉）
    """

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """引擎核心请求数据结构。

    封装了从前端发送到引擎核心的单个请求的所有信息。
    使用 msgspec.Struct 进行高效序列化，支持零拷贝传输。

    Attributes:
        request_id: 请求的唯一标识符
        prompt_token_ids: 输入 prompt 的 token ID 列表
        mm_features: 多模态特征列表（如图像、音频等）
        sampling_params: 采样参数（用于生成任务）
        pooling_params: 池化参数（用于池化任务）
        arrival_time: 请求到达时间戳
        lora_request: LoRA 适配器请求
        cache_salt: 缓存盐值，用于区分不同用户的缓存
        data_parallel_rank: 数据并行秩
        prompt_embeds: 提示嵌入张量（当使用嵌入输入时）
        client_index: 客户端索引，用于在扩展前端时将输出发送回正确的客户端
        current_wave: 当前请求波次，用于数据并行场景
        priority: 请求优先级
        trace_headers: 分布式追踪头
        resumable: 是否可恢复（用于流式输入）
        external_req_id: 用户提供的请求 ID
        reasoning_ended: 推理是否已结束
    """
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None = None

    # 客户端的索引，用于在扩展前端场景下将输出发送回同一客户端
    client_index: int = 0

    # 在数据并行场景下，指示该请求属于哪个波次，
    # 用于覆盖一种竞态条件：请求发送完成但波次完成通知尚未收到
    current_wave: int = 0
    priority: int = 0

    trace_headers: Mapping[str, str] | None = None
    resumable: bool = False

    # 用户提供的请求 ID。该字段在内部设置，
    # 从最初分配给 request_id 字段的提供值复制而来，
    # 参见 InputProcessor.assign_request_id()。
    # 用于输出中并支持 abort(req_id, internal=False)。
    external_req_id: str | None = None

    reasoning_ended: bool | None = None

    @property
    def params(self) -> SamplingParams | PoolingParams:
        """返回处理后的参数（采样或池化）。"""
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


class EngineCoreEventType(enum.IntEnum):
    """引擎核心请求事件类型枚举。

    QUEUED - 请求已加入队列
    SCHEDULED - 请求已被调度
    PREEMPTED - 请求已被抢占
    """

    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


class EngineCoreEvent(msgspec.Struct):
    """与请求关联的时间戳引擎核心事件。

    时间戳使用单调时间，用于引擎前端计算引擎核心事件之间的间隔。
    这些时间戳不应与其他进程的时间戳进行比较。

    Attributes:
        type: 事件类型
        timestamp: 事件发生的时间戳
    """

    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(
        cls, event_type: EngineCoreEventType, timestamp: float | None = None
    ) -> "EngineCoreEvent":
        """创建一个新的事件。

        Args:
            event_type: 事件类型
            timestamp: 时间戳，如果为 None 则使用当前单调时间

        Returns:
            新的 EngineCoreEvent 实例
        """
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """引擎核心输出数据结构。

    封装了引擎核心对单个请求的处理输出。

    Attributes:
        request_id: 请求的唯一标识符
        new_token_ids: 新生成的 token ID 列表
        new_logprobs: 新的 token logprobs（如果启用）
        new_prompt_logprobs_tensors: prompt logprobs 张量（如果启用）
        pooling_output: 池化输出张量（用于池化任务）
        finish_reason: 完成原因（如果请求已完成）
        stop_reason: 停止原因（字符串或整数）
        events: 事件列表
        kv_transfer_params: KV 传输参数
        trace_headers: 分布式追踪头
        num_cached_tokens: 命中前缀缓存的 token 数量（本地 + 外部）
        num_external_computed_tokens: 远程计算的 token 数量
        routed_experts: 路由专家信息
        num_nans_in_logits: logits 中 NaN 的数量（大于 0 表示输出损坏）
    """
    request_id: str
    new_token_ids: list[int]

    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None

    pooling_output: torch.Tensor | None = None

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[EngineCoreEvent] | None = None
    kv_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None
    # 命中前缀缓存的 token 数量（本地 + 外部）
    num_cached_tokens: int = 0
    # 远程计算的 token 数量（来自连接器的原始计数）
    num_external_computed_tokens: int = 0
    routed_experts: np.ndarray | None = None
    # logits 中 NaN 的数量
    # 大于 0 表示输出已损坏
    num_nans_in_logits: int = 0

    @property
    def finished(self) -> bool:
        """返回请求是否已完成。"""
        return self.finish_reason is not None


class UtilityOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """实用函数调用的输出数据结构。

    Attributes:
        call_id: 调用 ID
        failure_message: 失败消息（如果调用失败）
        result: 调用结果
    """
    call_id: int

    # 非 None 表示调用失败，result 应为 None
    failure_message: str | None = None
    result: UtilityResult | None = None


class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """引擎核心输出批量数据结构。

    封装了引擎核心在一次迭代中的批量输出。
    注意：当前采用行式布局（每个请求一个输出对象）

    Attributes:
        engine_index: 引擎索引
        outputs: 输出列表
        scheduler_stats: 调度器统计信息
        timestamp: 时间戳
        utility_output: 实用函数输出
        finished_requests: 已完成的请求 ID 集合
        wave_complete: 波次完成通知（数据并行场景）
        start_wave: 启动新波次通知（数据并行场景）
    """
    # 注意：可以考虑更紧凑的布局，例如列式布局

    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0

    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None

    # 在数据并行场景下，用于信号当前请求波次已完成且引擎已暂停
    wave_complete: int | None = None
    # 在数据并行场景下，用于信号接收到"旧"波次的请求，
    # 因此需要在其他引擎中启动新波次
    start_wave: int | None = None

    def __post_init__(self):
        """初始化时间戳（如果未设置）。"""
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


class EngineCoreRequestType(enum.Enum):
    """引擎核心请求类型枚举。

    定义为十六进制字节字符串，因此可以通过 socket 发送而无需单独的编码步骤。

    ADD - 添加新请求
    ABORT - 中止请求
    START_DP_WAVE - 启动数据并行波次
    UTILITY - 实用函数调用
    EXECUTOR_FAILED - 执行器失败（仅在 EngineCoreProc 内部使用）
    WAKEUP - 唤醒信号（用于在关闭期间唤醒 input_queue.get()）
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # EngineCoreProc 内部使用的哨兵
    EXECUTOR_FAILED = b"\x04"
    # 用于在关闭期间唤醒 input_queue.get() 的哨兵
    WAKEUP = b"\x05"


class ReconfigureDistributedRequest(msgspec.Struct):
    """重新配置分布式请求数据结构。

    用于在弹性专家并行（Elastic EP）场景中重新配置数据并行参数。

    Attributes:
        new_data_parallel_size: 新的数据并行大小
        new_data_parallel_rank: 新的数据并行秩
        new_data_parallel_rank_local: 新的本地数据并行秩
        new_data_parallel_master_ip: 新的数据并行主节点 IP
        new_data_parallel_master_port: 新的数据并行主节点端口
        new_data_parallel_master_port_list: 新的数据并行主节点端口列表
        coord_store_port: 协调器存储端口
    """
    new_data_parallel_size: int
    new_data_parallel_rank: int
    new_data_parallel_rank_local: int
    new_data_parallel_master_ip: str
    new_data_parallel_master_port: int
    new_data_parallel_master_port_list: list[int]
    coord_store_port: int


class ReconfigureRankType(enum.IntEnum):
    """重新配置分布式请求的秩类型枚举。

    KEEP_CURRENT_RANK - 保持当前秩
    SHUTDOWN_CURRENT_RANK - 关闭当前秩
    """

    KEEP_CURRENT_RANK = -1
    SHUTDOWN_CURRENT_RANK = -2
