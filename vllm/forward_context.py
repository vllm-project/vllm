# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：ForwardContext —— 一次模型 forward 的「全局上下文」总线。
# [CN] 为什么需要它：vLLM 的模型层被 torch.compile / CUDA graph 编译过，无法再靠函数
# [CN] 参数把 attention metadata、slot_mapping、DP 信息一路透传下去，于是改用「模块级
# [CN] 全局变量 + contextmanager」的方式，让任意深度的自定义算子都能拿到当前步的上下文。
# [CN] 链路位置：gpu_model_runner 准备 metadata → set_forward_context(...) 建立上下文 →
# [CN] 模型各层（attention / MoE / LoRA）通过 get_forward_context() 取用 → 退出时恢复。
# [CN] 与 v1/sample/metadata.py 的 SamplingMetadata 区分：
# [CN]   ForwardContext    = 模型前向阶段（attention metadata、slot_mapping、图模式…）；
# [CN]   SamplingMetadata  = 采样阶段（温度、top_p、惩罚、logprobs…）。
# [CN] 三个最容易看错的点：
# [CN]   1. attn_metadata / slot_mapping 在 DBO（双微批）下是「长度为 2 的 list」而非
# [CN]      dict，取值前必须先判断当前的 ubatch 索引，否则会直接取错微批。
# [CN]   2. _forward_context 是模块级全局量，靠 override_forward_context 用
# [CN]      try/finally 精确还原 —— 它不是线程安全的，一层 forward 内不可并发重入。
# [CN]   3. all_moe_layers / moe_layer_index 是为绕开 torch.compile 冷启动硬编码字符串
# [CN]      的权宜之计，依赖「自定义算子按层序执行且不被重排」这个假设。
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ubatch_utils import UBatchSlices

logger = init_logger(__name__)

# [CN] 下面几个模块级变量服务于「按 batch size 统计 forward 耗时」的可观测性功能：
# [CN] 由 VLLM_LOG_BATCHSIZE_INTERVAL 控制开关与打印间隔；batchsize_forward_time 以
# [CN] batch size 为 key 累积每次耗时，到点后打中位数。属纯诊断路径，不影响推理语义。
track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


# [CN] BatchDescriptor：CUDA graph 的「分发键」（frozen 保证可哈希、可作 dict key）。
# [CN] 反直觉点：字段要尽量少，每多一个字段就会让可复用的图数量按笛卡尔积爆炸；
# [CN] 所以它只描述「padding 后的形状」而不描述具体内容。
@dataclass(frozen=True)
class BatchDescriptor:
    """
    Batch descriptor for cudagraph dispatching. We should keep the num of
    items as minimal as possible to properly and uniquely describe the padded
    batch for cudagraph.
    """

    # [CN] num_tokens：padding 之后的 token 总数，决定要 replay 哪一张图。
    num_tokens: int
    num_reqs: int | None = None
    """
    Number of requests in the batch. Can be None for PIECEWISE cudagraphs where
    the cudagraphs can handle any number of requests.
    """
    # [CN] uniform：批内所有请求 token 数是否一致。True 时可以选用更特化（更快）的核。
    uniform: bool = False
    """
    True if all the requests in the batch have the same number of tokens.
    """
    has_lora: bool = False
    """
    Whether this batch has active LoRA adapters.
    """
    # [CN] num_active_loras：本批不同 LoRA 适配器的个数。开启
    # [CN] cudagraph_specialize_lora_count 后会按该值分别抓图，因为 fused_moe_lora 等核的
    # [CN] grid size 依赖它 —— 图一旦抓完形状就固定了，不能运行时再变。
    num_active_loras: int = 0
    """
    Number of distinct active LoRA adapters in this batch.
    When cudagraph_specialize_lora_count is enabled, separate CUDA graphs
    are captured for each num_active_loras value. This allows kernels
    (like fused_moe_lora) whose grid size depends on num_active_loras
    to be properly captured.
    """


# [CN] 把「各 DP rank 的 token 数」换算成序列并行（SP）视角下的每片 token 数：
# [CN] 先向上取整到 sp_size 的倍数，再 repeat_interleave 展开回每个 SP rank 一份。
# [CN] 这样 MoE 才能知道输入在 DP × SP 两个维度上分别被切成多长。
def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor, sequence_parallel_size: int
) -> list[int]:
    sp_tokens = (
        num_tokens_across_dp_cpu + sequence_parallel_size - 1
    ) // sequence_parallel_size

    sp_tokens = sp_tokens.repeat_interleave(sequence_parallel_size)
    return sp_tokens.tolist()


@dataclass
class DPMetadata:
    # [CN] DPMetadata：数据并行下的「各 rank token 数」视图，供 MoE 做 all-to-all 分发。
    # [CN] 注意名字里的 cpu：这张张量刻意留在 CPU 上（int32），因为它要参与 Python 侧的
    # [CN] 形状计算与断言；放 GPU 上每次读都要同步，反而更慢。
    num_tokens_across_dp_cpu: torch.Tensor

    # [CN] local_sizes：chunked_sizes / sp_local_sizes 上下文期间才有效的临时分片长度，
    # [CN] 退出上下文即被置回 None；因此读取它的代码必须在上下文内部执行。
    # NOTE: local_sizes should only be set by the chunked_sizes context manager
    local_sizes: list[int] | None = None

    @staticmethod
    # [CN] 构造 DPMetadata 的工厂方法，带一组「DP / SP-MoE 前提」断言：
    # [CN] 只有 data_parallel_size > 1 或启用 sequence_parallel_moe 且模型确为 MoE 时
    # [CN] 才会构造；并要求 num_tokens_across_dp[dp_rank] 与本 rank 的 batchsize 相等。
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata":
        assert num_tokens_across_dp_cpu is not None
        assert (
            parallel_config.data_parallel_size > 1
            or parallel_config.use_sequence_parallel_moe
        )
        assert parallel_config.is_moe_model is not False
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert num_tokens_across_dp_cpu[dp_rank] == batchsize, (
            f"{num_tokens_across_dp_cpu[dp_rank]} {batchsize}"
        )
        return DPMetadata(num_tokens_across_dp_cpu)

    @contextmanager
    # [CN] sp_local_sizes：与 chunked_sizes 同构但**不做分块**的上下文，用于纯粹的
    # [CN] 序列并行场景 —— 只把 local_sizes 设为各 SP 片的 token 数。
    def sp_local_sizes(self, sequence_parallel_size: int):
        """
        Context manager for setting self.local_sizes. Same as self.chunked_sizes
        but without any chunking.
        """
        self.local_sizes = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu, sequence_parallel_size
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        assert self.local_sizes is not None
        return self.local_sizes

    # Get the cumulative tokens across sequence parallel ranks.
    # In this case the input to the MoEs will be distributed w.r.t both
    # DP and TP rank.
    # When sp_size==1, this is just the cumulative num tokens across DP.
    # [CN] 返回跨 SP rank 的**累积** token 数（cumsum），MoE 用它算每个 rank 的输入偏移。
    # [CN] sp_size == 1 时退化为「跨 DP 的累积 token 数」。
    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        num_tokens_across_sp_cpu = (
            self.num_tokens_across_dp_cpu - 1 + sp_size
        ) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(sp_size)
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)


@dataclass
class ForwardContext:
    # [CN] ForwardContext：一次 forward 的动态上下文（每步重建）。
    # [CN] 字段分三类：静态配置（no_compile_layers）、每步动态量（attn_metadata /
    # [CN] slot_mapping / dp_metadata / batch_descriptor / ubatch_slices / is_padding）、
    # [CN] 编译与图模式相关开关（cudagraph_runtime_mode / skip_compiled / all_moe_layers）。
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    # [CN] attn_metadata：layer_name -> 该层 attention metadata。
    # [CN] **关键歧义**：DBO（双微批）场景下它是长度 2 的 list，一个元素对应一个微批，
    # [CN] 消费方必须先用 ubatch_slices 确定当前微批下标再去索引，不能直接当 dict 用。
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    """
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each
    attention layer to its attention metadata
    Type List[Dict[str, AttentionMetadata]] for DBO. List of size two, one
    for each microbatch.
    Set dynamically for each forward pass
    """
    # [CN] dp_metadata：非 DP / 非 SP-MoE 时为 None，MoE 分发逻辑需先判空。
    # set dynamically for each forward pass
    dp_metadata: DPMetadata | None = None
    # determine the cudagraph style at runtime to be FULL, PIECEWISE, or NONE.
    # by default NONE, no cudagraph is used.
    # [CN] cudagraph_runtime_mode：运行时决定本步是否走图、走哪种图（FULL / PIECEWISE /
    # [CN] NONE）。默认 NONE。它与 batch_descriptor 配合选出具体要 replay 的那张图。
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    # [CN] batch_descriptor：本步的图分发键（见 BatchDescriptor）。仅当运行时模式不是
    # [CN] NONE 时才需要；为 None 时图包装层会跳过 replay，直接走 eager。
    batch_descriptor: BatchDescriptor | None = None

    # [CN] ubatch_slices：DBO 微批切片信息；非 None 表示本步被切成两个微批，
    # [CN] 同时意味着 attn_metadata / slot_mapping 是 list 形态。
    ubatch_slices: UBatchSlices | None = None

    # Boolean mask over the token axis: True for padding rows that are not real
    # tokens. Consumers can use it to skip work for padded tokens. None when
    # the producer does not set it.
    # [CN] is_padding：沿 token 轴的 bool 掩码，True=该行是 padding 而非真实 token。
    # [CN] 消费方可用它跳过 padding 行的计算；None 表示生产者没有提供该信息。
    is_padding: torch.Tensor | None = None

    # If True, bypass the compiled model call, e.g. by using .forward() directly
    # [CN] skip_compiled：True 时绕过编译后的模型调用、直接走 .forward()。
    # [CN] 主要用于 profile / 调试路径，避免编译产物干扰测量。
    skip_compiled: bool = False

    # For torch.compile cold start times, we need to avoid hard-coding
    # any strings into the graph. Right now, the vllm.moe_forward
    # and vllm.moe_forward_shared custom operators hard-code strings into
    # the graph.
    #
    # The workaround is to store a list of the strings that each of those
    # custom ops needs in the ForwardContext (all_moe_layers)
    # as well as a counter (moe_layer_index).
    # The ForwardContext object is alive for the duration of the forward pass.
    # When the custom op needs a layer string, get the next string
    # from all_moe_layers and increment the counter.
    #
    # This assumes that the custom operators will always be executed in
    # order and that torch.compile will not try to reorder these
    # operations with respect to each other.
    #
    # TODO(https://github.com/vllm-project/vllm/issues/31985):
    # There are longer-term solutions, like unwrapping the moe custom operator,
    # that aren't ready yet.
    # We could also treat the string as a "symbolic input" to the graph but
    # the PyTorch-side bits for that aren't ready yet either.
    #
    # If this value is None (like in some tests), then we end up baking the string
    # into the graph. Otherwise, the moe custom ops will pop a string from this list.
    # [CN] all_moe_layers / moe_layer_index：为消除 torch.compile 冷启动时的字符串硬编码。
    # [CN] 做法是把所有 MoE 层名预先存进上下文，自定义算子每次按需 pop 一个并递增计数器；
    # [CN] 这依赖「算子按层序执行且编译期不重排」的假设。为 None 时退回硬编码字符串。
    all_moe_layers: list[str] | None = None
    moe_layer_index: int = 0

    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    # [CN] 唯一校验：cudagraph_runtime_mode 必须是合法的**运行时**模式（例如不能把
    # [CN] FULL_AND_PIECEWISE 这种「编译期配置」当成运行时模式传进来）。
    def __post_init__(self):
        assert self.cudagraph_runtime_mode.is_valid_runtime_mode(), (
            f"Invalid cudagraph runtime mode: {self.cudagraph_runtime_mode}"
        )


# [CN] 模块级全局上下文。注意这不是线程安全设计：一层 forward 期间靠
# [CN] override_forward_context 的 try/finally 精确还原，不可并发重入。
_forward_context: ForwardContext | None = None


# [CN] 取当前上下文。若未设置会直接 assert 失败 —— 也就是说任何依赖它的自定义算子
# [CN] 都必须运行在 set_forward_context 包裹的范围内。
def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


# [CN] 非断言版的探测接口：供「有没有上下文都能工作」的代码分支使用。
def is_forward_context_available() -> bool:
    return _forward_context is not None


# [CN] 纯构造 ForwardContext 的工厂（不负责安装到全局）。
# [CN] slot_mapping 缺省为 {}；all_moe_layers 只在开启 fast_moe_cold_start 时才注入
# [CN] 编译期静态层名列表，否则为 None。
def create_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
    is_padding: torch.Tensor | None = None,
):
    if vllm_config.compilation_config.fast_moe_cold_start:
        all_moe_layers = vllm_config.compilation_config.static_all_moe_layers
    else:
        all_moe_layers = None

    return ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        all_moe_layers=all_moe_layers,
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping or {},
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
        skip_compiled=skip_compiled,
        additional_kwargs=additional_kwargs or {},
        is_padding=is_padding,
    )


@contextmanager
# [CN] 直接把已有上下文装到全局（保存并在 finally 中还原旧值）。
# [CN] 与 set_forward_context 的区别：后者负责「构造 + 安装」，这里只做「安装」。
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager
# [CN] 整个推理链路最常用的入口：构造 ForwardContext 并在 with 期间安装到全局。
# [CN] 这里集中注入了每个模型 forward 都要做的公共逻辑：DP 协调、图分发键补全、
# [CN] 平台额外上下文、batch size 耗时统计。
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
    is_padding: torch.Tensor | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    # [CN] 只有开启了统计开关、且确实有 attn_metadata（排除纯 profile 的空跑）才计时。
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()

    dp_metadata: DPMetadata | None = None
    if (
        (
            vllm_config.parallel_config.data_parallel_size > 1
            or vllm_config.parallel_config.use_sequence_parallel_moe
        )
        and vllm_config.parallel_config.is_moe_model is not False
        and (attn_metadata is not None or num_tokens is not None)
    ):
        # If num_tokens_across_dp hasn't already been initialized, then
        # initialize it here. Both DP padding and Microbatching will be
        # disabled.
        if (
            num_tokens_across_dp is None
            and vllm_config.parallel_config.data_parallel_size > 1
        ):
            assert ubatch_slices is None
            assert num_tokens is not None
            _, num_tokens_across_dp, _ = coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens,
                parallel_config=vllm_config.parallel_config,
                allow_microbatching=False,
            )
            assert num_tokens_across_dp is not None
        elif num_tokens_across_dp is None:
            assert num_tokens is not None
            num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int32)
        dp_metadata = DPMetadata.make(
            vllm_config.parallel_config, num_tokens or 0, num_tokens_across_dp
        )

    # [CN] 便利补全：走了图模式且给了 num_tokens 时，若调用方没传 batch_descriptor，
    # [CN] 这里就地造一个只含 num_tokens 的最小描述符；即便与图包装层期望的不匹配，
    # [CN] 也只会退化为不命中缓存，不会有正确性问题。
    # Convenience: if cudagraph is used and num_tokens is given, we can just
    # create a batch descriptor here if not given (there's no harm since if it
    # doesn't match in the wrapper it'll fall through).
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)

    # [CN] 平台钩子：让不同后端（CUDA / ROCm / TPU …）往上下文里塞自己的额外字段，
    # [CN] 保持上层模型代码与平台无关。
    additional_kwargs = current_platform.set_additional_forward_context(
        attn_metadata=attn_metadata,
        vllm_config=vllm_config,
        dp_metadata=dp_metadata,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
    )

    forward_context = create_forward_context(
        attn_metadata,
        vllm_config,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
        slot_mapping,
        additional_kwargs,
        skip_compiled,
        is_padding=is_padding,
    )

    try:
        # [CN] 真正的安装点。finally 块里做耗时统计：先 synchronize 再取 perf_counter，
        # [CN] 因为当前是同步调度，在这里插入同步点不会影响下一步的调度。
        with override_forward_context(forward_context):
            yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            synchronize = current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append((now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(
                        (
                            "Batchsize forward time stats "
                            "(batchsize, count, median_time(ms)): %s"
                        ),
                        forward_stats,
                    )
