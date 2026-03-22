# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""调度器输出模块。

本模块定义了调度器输出相关的数据类，负责：
- 封装新请求的数据
- 封装缓存请求的数据
- 封装调度器输出
- 封装文法输出（结构化输出）

主要类：
- NewRequestData: 新请求数据
- CachedRequestData: 缓存请求数据
- SchedulerOutput: 调度器输出
- GrammarOutput: 文法输出
"""

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@dataclass
class NewRequestData:
    """新请求数据类。

    封装首次调度的请求的数据。我们将请求的数据缓存在每个 worker
    进程中，因此不需要在每个调度步骤重新发送。

    Attributes:
        req_id: 请求的唯一标识符
        prompt_token_ids: prompt token ID 列表
        mm_features: 多模态特征列表
        sampling_params: 采样参数（用于生成任务）
        pooling_params: 池化参数（用于池化任务）
        block_ids: 每个 KV 缓存组的块 ID 元组
        num_computed_tokens: 已计算的 token 数量
        lora_request: LoRA 请求
        prompt_embeds: prompt 嵌入（可选）
        prefill_token_ids: 预填充 token ID（仅用于 v2 model runner）
    """

    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None

    # 仅用于 v2 model runner
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        """从 Request 对象创建 NewRequestData。

        Args:
            request: 请求对象
            block_ids: 块 ID 元组
            prefill_token_ids: 预填充 token ID（可选）

        Returns:
            NewRequestData 对象
        """
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        """返回请求数据的字符串表示。"""
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # __repr__ 的版本，隐藏 prompt 数据
    def anon_repr(self) -> str:
        """返回匿名版本的字符串表示（隐藏 token 数据）。

        Returns:
            匿名版本的字符串表示
        """
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@dataclass
class CachedRequestData:
    """缓存请求数据类。

    封装之前已调度过的请求的数据。由于请求的数据已经缓存在
    worker 进程中，我们只发送差异以最小化通信开销。

    Attributes:
        req_ids: 请求 ID 列表
        resumed_req_ids: 恢复的请求 ID 集合
            对于不在 resumed_req_ids 中的请求 ID，new_block_ids 将附加到
            请求的块 ID 后面。对于在集合中的请求，new_block_ids 将用作
            请求的块 ID，而不是附加到现有块 ID。
        new_token_ids: 新 token ID 列表（仅用于流水线并行）
            当不使用 PP 时，此列表为空。
        all_token_ids: 所有 token ID 字典
            对于上一步未调度的请求，将 token ID 传播给连接器。
            不包含上一步已调度的请求。
        new_block_ids: 新块 ID 列表
        num_computed_tokens: 每个请求已计算的 token 数量
        num_output_tokens: 每个请求已输出的 token 数量
    """

    req_ids: list[str]
    # 对于不在 resumed_req_ids 中的请求 ID，new_block_ids 将附加到
    # 请求的块 ID 后面。对于在集合中的请求，new_block_ids 将用作
    # 请求的块 ID，而不是附加到现有块 ID。
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids 仅用于流水线并行。
    # 当不使用 PP 时，new_token_ids 将为空。
    new_token_ids: list[list[int]]
    # 对于上一步未调度的请求，将 token ID 传播给连接器。
    # 不包含上一步已调度的请求。
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    # 隐藏 token ID 的版本
    def anon_repr(self) -> str:
        """返回匿名版本的字符串表示（隐藏 token 数据）。

        Returns:
            匿名版本的字符串表示
        """
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        """返回请求数据的字符串表示。"""
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        """返回请求数量。

        Returns:
            请求数量
        """
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """缓存 req_id 到 num_output_tokens 的映射，用于 O(1) 查找。

        这个 cached_property 是安全的，因为 CachedRequestData 实例
        在每个调度迭代中都是新创建的，并且在计算过程中不会被修改。

        Returns:
            req_id 到 num_output_tokens 的字典
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        """检查请求是否处于上下文阶段（尚未生成输出）。

        Args:
            req_id: 请求 ID

        Returns:
            如果请求处于上下文阶段则返回 True
        """
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        """创建空的 CachedRequestData。

        Returns:
            空的 CachedRequestData 对象
        """
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class SchedulerOutput:
    """调度器输出数据类。

    封装调度器的输出结果，包含所有调度决策的信息。

    Attributes:
        scheduled_new_reqs: 首次调度的请求列表
            我们缓存每个 worker 进程中的请求数据，因此不需要在
            每个调度步骤重新发送。
        scheduled_cached_reqs: 之前已调度过的请求列表
            由于请求数据已经缓存在 worker 进程中，我们只发送差异
            以最小化通信开销。
        num_scheduled_tokens: req_id -> num_scheduled_tokens 字典
            每个请求调度的 token 数量。
        total_num_scheduled_tokens: 所有请求调度的 token 总数
            等于 sum(num_scheduled_tokens.values())
        scheduled_spec_decode_tokens: req_id -> spec_token_ids 字典
            如果请求没有任何 spec decode token，则不会包含在字典中。
        scheduled_encoder_inputs: req_id -> encoder input indices 字典
            需要处理的编码器输入索引。
            例如，如果请求有 [0, 1]，可能表示视觉编码器需要处理
            请求的第 0 个和第 1 个图像。
        num_common_prefix_blocks: 每个 KV 缓存组的公共前缀块数量
            可用于 cascade attention。
        finished_req_ids: 在上一步和当前步之间完成的请求 ID 集合
            用于通知 worker 关于已完成的请求，以便它们可以释放
            这些请求的缓存状态。
        free_encoder_mm_hashes: 要从编码器缓存释放的 mm_hash 字符串列表
        preempted_req_ids: 在此步骤中被抢占的请求 ID 集合
            仅用于 v2 model runner。
        has_structured_output_requests: 是否有任何调度的请求使用结构化输出
            仅在异步调度情况下设置。
        pending_structured_output_tokens: 调度的请求是否有所有需要的输出 token
            用于执行文法掩码计算。
        num_invalid_spec_tokens: 用于调整接受率计算的无效 spec token 数量
        kv_connector_metadata: KV 缓存连接器元数据
        ec_connector_metadata: EC 缓存连接器元数据
        new_block_ids_to_zero: 在此调度步骤中从池中新分配的块 ID 列表
            worker 在块使用之前清零相应的 GPU 内存，防止过时的
            NaN/数据污染注意力或 SSM 计算。
    """

    # 首次调度的请求列表
    scheduled_new_reqs: list[NewRequestData]
    # 之前已调度过的请求列表
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    num_scheduled_tokens: dict[str, int]
    # 所有请求调度的 token 总数
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> 需要处理的编码器输入索引
    scheduled_encoder_inputs: dict[str, list[int]]
    # 每个 KV 缓存组的公共前缀块数量
    num_common_prefix_blocks: list[int]

    # 在上一步和当前步之间完成的请求 ID 集合
    finished_req_ids: set[str]
    # 要从编码器缓存释放的 mm_hash 字符串列表
    free_encoder_mm_hashes: list[str]

    # 在此步骤中被抢占的请求 ID 集合
    # 仅用于 v2 model runner
    preempted_req_ids: set[str] | None = None

    # 是否有任何调度的请求使用结构化输出
    # 仅在异步调度情况下设置
    has_structured_output_requests: bool = False

    # 调度的请求是否有所有需要的输出 token 用于执行文法掩码计算
    pending_structured_output_tokens: bool = False

    # 用于调整接受率计算
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV 缓存连接器元数据
    kv_connector_metadata: KVConnectorMetadata | None = None

    # EC 缓存连接器元数据
    ec_connector_metadata: ECConnectorMetadata | None = None

    # 在此调度步骤中从池中新分配的块 ID 列表
    new_block_ids_to_zero: list[int] | None = None

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        """创建空的 SchedulerOutput。

        Returns:
            空的 SchedulerOutput 对象
        """
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    """文法输出数据类。

    封装结构化输出的文法位掩码信息。

    Attributes:
        structured_output_request_ids: 结构化输出请求的 ID 列表
        grammar_bitmask: 位掩码，顺序与 structured_output_request_ids 对应
            类型为 numpy int32 数组
    """

    # 结构化输出请求的 ID 列表
    structured_output_request_ids: list[str]
    # 位掩码，顺序与 structured_output_request_ids 对应
    grammar_bitmask: "npt.NDArray[np.int32]"
