# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：引擎内部的输出契约（ModelRunnerOutput 及其卫星结构）
# [CN] 职责：定义「worker 每步跑完要交给 scheduler 什么东西」，是 V1 前后段的唯一数据接口。
# [CN] 链路：GPUModelRunner -> ModelRunnerOutput -> Scheduler.update_from_output()
# [CN]       -> EngineCoreOutput -> output_processor -> 用户侧的 RequestOutput
# [CN] 成对出现的两种形态（名字一个 Lists 一个 Tensors）：
# [CN]   *Lists   —— CPU 侧 numpy / list，可以直接跨进程序列化
# [CN]   *Tensors —— GPU 侧 torch 张量，仅供单个 worker 内部使用
# [CN] 贯穿全文的两个设计约束：
# [CN]   1. 要跨进程就必须是 list / numpy：ModelRunnerOutput 会被 msgpack 序列化，
# [CN]      序列化一个 torch.Tensor 代价极高，所以对外一律换成 list。
# [CN]   2. 张量转宿主必须 non_blocking 且目标是 pinned memory，
# [CN]      否则达不到「异步调度」想要的 GPU / CPU 重叠。

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import numpy as np
import torch

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorWorkerMetadata
    from vllm.distributed.kv_events import KVConnectorKVEvents
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
else:
    KVConnectorStats = object
    KVConnectorWorkerMetadata = object
    KVConnectorKVEvents = object
    ECConnectorWorkerMetadata = object


# [CN] logprobs 的 CPU 形态。所有行首维都是「扁平化的 num_reqs x num_generated_tokens」，
# [CN] 因此需要额外的 cu_num_generated_tokens 才能还原出每个请求属于哪一段。

class LogprobsLists(NamedTuple):
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    # [CN] 每个位置返回 max_num_logprobs+1 个 token：+1 的那位是实际被采中的 token
    # [CN] （它可能不在 top-k 里，比如低概率被采样到时）。

    logprob_token_ids: np.ndarray
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: np.ndarray
    # [num_reqs x num_generated_tokens]
    sampled_token_ranks: np.ndarray
    # [num_reqs]
    # Used for slicing the logprobs in cases like speculative
    # decoding where the number of generated tokens may be
    # different for each request.
    # [CN] CSR 风格的前缀和：第 i 个请求的数据落在 [cu[i], cu[i+1])。没有它就无法反解扁平数组。

    cu_num_generated_tokens: list[int] | None = None

    # [CN] 按请求切出一段。req_idx 在有 cu 表时要先换算成真实起点 ——
    # [CN] 投机解码下每个请求本步生成的 token 数不一样，起点不能用 req_idx * 固定步长推算。

    def slice_request(self, req_idx: int, num_positions: int):
        if self.cu_num_generated_tokens is not None:
            req_idx = self.cu_num_generated_tokens[req_idx]
        end_idx = req_idx + num_positions
        return LogprobsLists(
            self.logprob_token_ids[req_idx:end_idx],
            self.logprobs[req_idx:end_idx],
            self.sampled_token_ranks[req_idx:end_idx],
            None,
        )


# [CN] CSR 压缩布局的采样支撑集。一步一个位置时 offsets 为 None，多位置时才需要 offsets。

class SamplingMaskLists(NamedTuple):
    """CSR sampling masks; a step slice holds one position (``offsets=None``)."""

    # [num_kept_tokens]
    token_ids: np.ndarray
    # [num_positions + 1], or None for a single position
    offsets: np.ndarray | None = None
    # Unused with one position per request; kept for the wire layout.
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, req_idx: int, num_positions: int) -> "SamplingMaskLists":
        assert num_positions == 1 and self.offsets is not None
        return SamplingMaskLists(
            self.token_ids[self.offsets[req_idx] : self.offsets[req_idx + 1]]
        )

    # [CN] CSR -> 嵌套 list 的解压。这是给用户看的形态给用户看的形态，代价是把 numpy 摊成 Python 对象。

    def to_nested_list(self) -> list[list[int]]:
        token_ids = self.token_ids.tolist()
        if self.offsets is None:
            return [token_ids]
        offsets = self.offsets.tolist()
        return [token_ids[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


# [CN] logprobs 的 GPU 形态。三个张量都在设备上，转 CPU 形态由 to_cpu_nonblocking 负责。

class LogprobsTensors(NamedTuple):
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs x num_generated_tokens]
    # [CN] 实际采中的 token 在词表中的排名（rank）。
    # [CN] rank 与 logprob 是两个视角：前者便于算 perplexity，后者用于展示概率。

    selected_token_ranks: torch.Tensor
    # [num_reqs + 1]
    cu_num_generated_tokens: list[int] | None = None
    # [num_reqs + 1]. Set instead of cu_num_generated_tokens when the
    # boundaries only exist on device (adaptive verification); rides along on
    # the async D2H copy.
    # [CN] 自适应校验场景下分段边界只存在于设备上（长度由 device 决定，CPU 无法预知），
    # [CN] 因此让这张 device 张量搭异步 D2H 的车一起下来。

    cu_num_generated_tokens_tensor: torch.Tensor | None = None

    # [CN] GPU ->numpy 的最终落地。注意这里 .cpu() 会同步：只有在测试路径上才这样直接调用。

    def tolists(self, cu_num_generated_tokens: list[int] | None = None):
        if cu_num_generated_tokens is None:
            if self.cu_num_generated_tokens_tensor is not None:
                cu_num_generated_tokens = self.cu_num_generated_tokens_tensor.tolist()
            else:
                cu_num_generated_tokens = self.cu_num_generated_tokens
        return LogprobsLists(
            self.logprob_token_ids.cpu().numpy(),
            self.logprobs.cpu().numpy(),
            self.selected_token_ranks.cpu().numpy(),
            cu_num_generated_tokens,
        )

    # [CN] 异步搬数据：non_blocking=True 只有在目标是 pinned memory 时才真的异步，
    # [CN] 否则会退化成同步拷贝（不报错，只是没收益）。cu 张量也要一起搬。

    def to_cpu_nonblocking(self) -> "LogprobsTensors":
        if self.logprob_token_ids.device.type == "cpu":
            return self
        cu_tensor = self.cu_num_generated_tokens_tensor
        if cu_tensor is not None:
            cu_tensor = cu_tensor.to("cpu", non_blocking=True)
        return LogprobsTensors(
            self.logprob_token_ids.to("cpu", non_blocking=True),
            self.logprobs.to("cpu", non_blocking=True),
            self.selected_token_ranks.to("cpu", non_blocking=True),
            self.cu_num_generated_tokens,
            cu_tensor,
        )

    # [CN] 按 bool mask 过滤。assert 要求 cu 表为空 —— 带了分段表再 filter 会让分段信息失效。

    def filter(self, mask: torch.Tensor) -> "LogprobsTensors":
        """Filter the logprobs tensors with the given bool mask."""
        assert self.cu_num_generated_tokens is None, (
            "filter can't be used with cu_num_generated_tokens"
        )
        assert self.cu_num_generated_tokens_tensor is None, (
            "filter can't be used with cu_num_generated_tokens_tensor"
        )
        return LogprobsTensors(
            self.logprob_token_ids[mask],
            self.logprobs[mask],
            self.selected_token_ranks[mask],
        )

    @staticmethod
    # [CN] 多 chunk 拼接时把分段表交给调用方重建：
    # [CN] 各 chunk 自己的分段边界在拼完之后就失去意义了。

    def cat(
        tensors: Sequence["LogprobsTensors"],
        cu_num_generated_tokens: list[int] | None = None,
    ) -> "LogprobsTensors":
        """Concatenate flattened logprob tensors."""
        assert tensors
        assert cu_num_generated_tokens is not None or all(
            tensor.cu_num_generated_tokens is None for tensor in tensors
        )
        if len(tensors) == 1:
            tensor = tensors[0]
            if cu_num_generated_tokens is None:
                return tensor
            return tensor._replace(cu_num_generated_tokens=cu_num_generated_tokens)
        # The multi-chunk path rebuilds boundaries from the CPU layout, which
        # the device-only boundaries of adaptive verification never use.
        assert all(tensor.cu_num_generated_tokens_tensor is None for tensor in tensors)
        return LogprobsTensors(
            logprob_token_ids=torch.cat(
                [tensor.logprob_token_ids for tensor in tensors]
            ),
            logprobs=torch.cat([tensor.logprobs for tensor in tensors]),
            selected_token_ranks=torch.cat(
                [tensor.selected_token_ranks for tensor in tensors]
            ),
            cu_num_generated_tokens=cu_num_generated_tokens,
        )

    @staticmethod
    # [CN] 预分配 CPU 侧的固定缓冲。pin_memory 是为了让后续 non_blocking 拷贝真正异步。

    def empty_cpu(
        num_positions: int, num_tokens_per_position: int
    ) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""

        logprob_token_ids = torch.empty(
            (num_positions, num_tokens_per_position),
            dtype=torch.int32,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        logprobs = logprob_token_ids.new_empty(
            (num_positions, num_tokens_per_position),
            dtype=torch.float32,
            pin_memory=PIN_MEMORY,
        )
        selected_token_ranks = logprob_token_ids.new_empty(
            num_positions, pin_memory=PIN_MEMORY
        )
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


# [CN] MoE 路由记录的设备快照。它必须在**私有副本**上做 D2H ——
# [CN] 如果直接引用共享的 capturer / prepare-input 缓冲，下一步 forward 会覆盖掉还在飞行的数据。

class RoutedExpertsTensors(NamedTuple):
    """Device-side snapshot of routed experts data, pending async D2H.

    Produced by :class:`GPUModelRunner` at the end of each async-scheduled
    step. The copy stream waits on the default stream, then issues
    non-blocking D2H via :meth:`to_cpu_nonblocking` into a pinned CPU
    buffer; :class:`AsyncGPUModelRunnerOutput.get_output` synchronizes
    the copy before the scheduler reads it.

    Sliced to ``total_num_scheduled_tokens`` (step-level, across all
    requests — NOT per-request). Both ``routing_data`` and
    ``slot_mapping`` must be private clones when sourced from shared
    capturer / prepare-input buffers, so the next forward pass /
    ``_prepare_inputs`` on the default stream does not race with a
    D2H still pending on the copy stream.
    """

    # (num_scheduled_tokens, num_layers, num_experts_per_tok)
    routing_data: torch.Tensor
    # (num_scheduled_tokens,)
    slot_mapping: torch.Tensor

    # [CN] 注意这里的降级：目标 CPU 张量未 pinned 时会退化为同步拷贝。
    # [CN] 可以接受，因为同步发生在**专用的拷贝流**上，不会拖住默认计算流。

    def to_cpu_nonblocking(self) -> "RoutedExpertsTensors":
        """Issue non-blocking D2H on the current stream.

        NOTE: ``non_blocking=True`` only delivers true overlap when the
        CPU target is pinned. The current fallback here allocates a
        new pageable CPU tensor per call, which silently degrades to a
        synchronous copy; acceptable because the sync happens on the
        dedicated copy stream, not the default stream.
        """
        if self.routing_data.device.type == "cpu":
            return self
        return RoutedExpertsTensors(
            self.routing_data.to("cpu", non_blocking=True),
            self.slot_mapping.to("cpu", non_blocking=True),
        )

    def tolists(self) -> "RoutedExpertsLists":
        """Convert to the numpy-backed form consumed by the scheduler.

        ``.cpu()`` is a no-op when the tensor is already on CPU, so this
        is cheap for the post-D2H case; for raw device tensors it will
        synchronously block, which is only reached in tests.
        """
        return RoutedExpertsLists(
            self.routing_data.cpu().numpy(),
            self.slot_mapping.cpu().numpy(),
        )


# [CN] 路由记录的 CPU 形态。首维是「本步全局调度的 token 数」而不是单请求的 token 数 ——
# [CN] 靠 slot_mapping 才能知道每一行属于哪个物理 KV slot。

class RoutedExpertsLists(NamedTuple):
    """CPU-side routed experts, the form :meth:`RoutedExpertsManager.store_batch`
    consumes.

    Batched per scheduler step: the leading dim is the number of tokens
    scheduled across all requests in this step (``total_num_scheduled_tokens``),
    not per-request tokens. ``slot_mapping[i]`` tells the scheduler which
    physical KV-cache slot row ``i`` of ``routing_data`` belongs to.
    """

    # (num_scheduled_tokens, num_layers, num_experts_per_tok)
    routing_data: np.ndarray
    # (num_scheduled_tokens,)
    slot_mapping: np.ndarray


# [num_reqs, <dynamic>]
# The shape of each element depends on the pooler used
# [CN] 池化输出可能是单张量、张量列表、或带 None 的列表 —— 取决于具体 pooler 的实现方式。

PoolerOutput: TypeAlias = torch.Tensor | list[torch.Tensor] | list[torch.Tensor | None]


# [CN] Sampler 的单步产出，只在单个 worker 内部流动（含 GPU 张量，不可序列化）。

@dataclass
class SamplerOutput:
    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    # [CN] 统一 pad 到本步最大生成数，空洞填 PLACEHOLDER_TOKEN_ID(-1)：
    # [CN] 投机解码下不同请求本步产出的 token 数天然不一致。

    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None


# [CN] KV connector（PD 分离）的单步产出。全部用 set / 标量便于后续聚合。

@dataclass
class KVConnectorOutput:
    # [req_ids]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    kv_connector_stats: KVConnectorStats | None = None
    kv_cache_events: KVConnectorKVEvents | None = None
    kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None
    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them
    # [CN] 加载失败的外部 KV 块。调度器需要让引用这些块的请求重算，而不是直接继续。

    invalid_block_ids: set[int] = field(default_factory=set)
    # Configuration describing how many finished sending/receiving
    # notifications should be expected for each request. This allows
    # handshake-based connectors like Nixl to update the KVOutputAggregator.
    # It captures a static setup info and should almost always remain constant
    # for a given connector after discovery. Default value entails no change.
    # [CN] 期望收到几份「完成」通知。Nixl 这类需要握手的 connector 靠它判断一轮是否收齐。

    expected_finished_count: int = 0

    # [CN] 「本步 connector 什么都没发生」的判定。空则不写进 ModelRunnerOutput，省一次序列化。

    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.kv_connector_stats
            and not self.kv_cache_events
            and not self.invalid_block_ids
            and not self.kv_connector_worker_meta
        )


# [CN] encoder cache connector 的单步产出，结构与 KV 侧基本一致，独立是因为协议不同。

@dataclass
class ECConnectorOutput:
    # [mm_hash]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    ec_connector_worker_meta: ECConnectorWorkerMetadata | None = None

    # [CN] EC 侧的空判定：比 KV 侧少两个字段，因为 encoder cache 没有 stats/events。

    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.ec_connector_worker_meta
        )


# [CN] 序列化友好性优先：能用 list 的地方绝不用 tensor，因为它每步都要跨进程传一次。

# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
# [CN] ModelRunnerOutput —— worker 交给 scheduler 的完整单步结果，本文件的核心。

@dataclass
class ModelRunnerOutput:
    # [num_reqs]
    # [CN] 本步参与的请求 id 列表，顺序与所有其他字段的行顺序严格对应 ——
    # [CN] 任何按位置索引的处理都必须以它为准。

    req_ids: list[str]
    # req_id -> index
    # [CN] 显式维护「请求 -> 本步下标」映射：auto_regressive 调度下各分片的下标会变化，
    # [CN] 不能假设它在整个输出处理过程中恒定。

    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: LogprobsLists | None = None

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    # [CN] prompt logprobs 按 request id 索引而不是按位置：它只在 prefill 那一步产生，
    # [CN] 且各请求长度不同，无法塞进统一的矩形张量。

    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(
        default_factory=dict
    )

    # [num_reqs, hidden_size]
    # [CN] 注意这里仍然是 torch 张量列表（唯一例外）：池化结果通常只在同步返回时要用，
    # [CN] 且张量本身不大，行列的带宽开销远小于一次额外的拷贝。

    pooler_output: list[torch.Tensor | None] | None = None

    kv_connector_output: KVConnectorOutput | None = None

    ec_connector_output: ECConnectorOutput | None = None

    # req_id -> num_nans_in_logits
    # [CN] 每个请求 logits 里 NaN 的数量。这是**诊断信息**而非正常路径数据 ——
    # [CN] 出现非零基本说明数值出了问题（如 dtype 溢出），值得立刻停下查。

    num_nans_in_logits: dict[str, int] | None = None

    # information related to cudagraph execution
    # [CN] CUDA Graph 的执行统计（是否命中、是否新捕获）。用于诊断图回放行为。

    cudagraph_stats: CUDAGraphStat | None = None

    # Per-step routed experts data captured by the worker.
    # ``routing_data`` shape: (num_scheduled_tokens, num_layers,
    #                         num_experts_per_tok); expert IDs as uint8/uint16.
    # ``slot_mapping`` shape: (num_scheduled_tokens,); physical KV-cache
    #                         slot for each row of routing_data.
    # ``num_scheduled_tokens`` is step-level (total across all requests
    # in this step), not per-request. The scheduler persists this into
    # its slot buffer via ``slot_buffer[slot_mapping] = routing_data``.
    # ``None`` when ``enable_return_routed_experts`` is off.
    # [CN] 未开启 enable_return_routed_experts 时为 None —— 不要假设这个字段总存在。

    routed_experts: RoutedExpertsLists | None = None

    # ``None`` when ``return_sampling_mask`` is off.
    # [CN] 未开启 return_sampling_mask 时为 None —— 结构化输出等场景才需要。

    sampling_masks: SamplingMaskLists | None = None

    @staticmethod
    # [CN] 返回一个「只有 connector 结果」的空输出。用共享空对象还是 copy，取决于是否要写它。

    def with_kv_conn_output_only(
        kv_connector_output: KVConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return ModelRunnerOutput containing the provided KVConnectorOutput,
        otherwise empty. Returns None if kv_connector_output is passed as None.
        """
        if kv_connector_output is None or kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    @staticmethod
    def with_ec_conn_output_only(
        ec_connector_output: ECConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return an otherwise-empty output carrying `ec_connector_output`."""
        return ModelRunnerOutput.with_ec_conn_output(
            EMPTY_MODEL_RUNNER_OUTPUT, ec_connector_output
        )

    @staticmethod
    # [CN] 共享的空输出不能被就地修改（其他调用方也持有它），所以这里 copy 后返回新对象 ——
    # [CN] 调用方必须使用返回值，忽略返回值就丢数据。

    def with_ec_conn_output(
        output: "ModelRunnerOutput",
        ec_connector_output: ECConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return `output` carrying `ec_connector_output`.

        The shared empty output is copied rather than written to, so callers
        must use the return value.
        """
        if ec_connector_output is None or ec_connector_output.is_empty():
            return output
        if output is EMPTY_MODEL_RUNNER_OUTPUT:
            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.ec_connector_output = ec_connector_output
        return output


# ModelRunnerOutput wrapper for async scheduling.
# [CN] 异步调度下的「结果的承诺」。get_output() 会阻塞直到神圣的两端同步完成，
# [CN] 且**每个对象只能调一次** —— 第二次拿不到东西。

class AsyncModelRunnerOutput(ABC):
    @abstractmethod
    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are ready, which
        might involve copying device tensors to the host.
        This method should only be called once per AsyncModelRunnerOutput.
        """
        pass


# [CN] 草稿模型的产出（投机解码专用）：只记 draft token id，不参与采样/概率统计。

@dataclass
class DraftTokenIds:
    # [num_reqs]
    # [CN] 草稿模型的请求列表。与主 ModelRunnerOutput 的请求集合相同但内容不同：
    # [CN] 这里装的是 draft token，不是真正输出到用户的 token。

    req_ids: list[str]
    # num_reqs x num_draft_tokens
    # [CN] 每个请求一份草稿 token 序列，长度取决于具体 draft 模型的配置。

    draft_token_ids: list[list[int]]


# [CN] encoder 实例从不采样，因此构造一个「账目齐全但没有生成数据」的占位输出，
# [CN] 让 scheduler 能正常地把这些请求走完并结束。

def make_empty_encoder_model_runner_output(
    scheduler_output: "SchedulerOutput",
) -> ModelRunnerOutput:
    """
    Create a ModelRunnerOutput stub that contains the correct
    per-request bookkeeping but no generated data yet.
    """
    if not scheduler_output.num_scheduled_tokens:
        return EMPTY_MODEL_RUNNER_OUTPUT

    # Convert to list so we get a deterministic, indexable sequence
    req_ids: list[str] = list(scheduler_output.num_scheduled_tokens.keys())

    # Give every request its own contiguous index
    req_id_to_index: dict[str, int] = {rid: idx for idx, rid in enumerate(req_ids)}

    # An encoder instance never samples, so it emits no tokens at all. The
    # scheduler finishes these requests once their prompt is fully encoded
    # (see `Scheduler.update_from_output`).
    sampled_token_ids: list[list[int]] = [[] for _ in req_ids]

    # Pooler outputs are not available yet ⇒ use None placeholders
    pooler_output: list[torch.Tensor | None] = [None for _ in req_ids]

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        pooler_output=pooler_output,
    )


# [CN] 全局共享的空输出单例。好处是零分配；代价是任何想写它的人都必须先 copy。

EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})
