# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""并行采样（Parallel Sampling）处理模块。

本模块实现了 vLLM 的并行采样功能，当用户请求生成多个候选输出（n > 1）时，
该模块负责管理父请求和子请求之间的关系，包括：
- 为每个子请求生成独立的采样参数
- 跟踪子请求的完成状态
- 聚合子请求的输出结果
"""

from copy import copy
from typing import cast

from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.metrics.stats import IterationStats


class ParentRequest:
    """并行采样请求的信息、状态和处理类。

    存储父请求 ID 和采样参数，支持生成子请求的采样参数。

    当用户请求生成多个候选输出（n > 1）时，系统会创建一个 ParentRequest
    来管理所有子请求。每个子请求会独立执行，但它们共享同一个父请求
    的上下文信息。

    Attributes:
        request_id: 父请求 ID
        external_req_id: 外部请求 ID（用户提供的 ID）
        sampling_params: 采样参数
        child_requests: 子请求 ID 集合
        output_aggregator: 用于聚合子请求输出的列表
        max_num_generation_tokens: 所有子请求中最大的生成 token 数
        cached_child_sampling_params: 缓存的子请求采样参数（用于优化）
    """

    request_id: str
    external_req_id: str
    sampling_params: SamplingParams

    # 用于跟踪子请求的完成情况
    child_requests: set[str]

    # 用于在非流式模式下聚合子请求的输出
    output_aggregator: list[CompletionOutput]

    # 用于跟踪所有子请求中最大的生成 token 数
    max_num_generation_tokens: int

    # 用于高效获取子请求采样参数
    cached_child_sampling_params: SamplingParams | None

    def __init__(self, request: EngineCoreRequest) -> None:
        """初始化 ParentRequest。

        Args:
            request: 引擎核心请求
        """
        assert request.external_req_id is not None
        sampling_params = request.params
        self.request_id = request.request_id
        self.external_req_id = request.external_req_id
        self.sampling_params = sampling_params

        self.child_requests = set()
        self.output_aggregator = (
            [cast(CompletionOutput, None)] * sampling_params.n
            if (sampling_params.output_kind == RequestOutputKind.FINAL_ONLY)
            else []
        )
        self.max_num_generation_tokens = 0
        self.cached_child_sampling_params = None

    def _get_child_sampling_params(
        self,
        index: int,
    ) -> SamplingParams:
        """高效获取子请求的 sampling_params。

        如果 sampling_params.seed 不为 None，则每个子请求需要一个
        唯一的 seed，因此需要创建父 sampling_params 的唯一克隆。

        Args:
            index: 在 n 个子请求中的索引

        Returns:
            子请求的 sampling_params 实例
        """
        seed = self.sampling_params.seed
        if self.cached_child_sampling_params:
            # 复用子请求 sampling_params 数据结构
            return self.cached_child_sampling_params
        # 构建子请求 sampling_params
        child_sampling_params = copy(self.sampling_params)
        child_sampling_params.n = 1
        if seed is None:
            # 缓存子请求 sampling_params 供后续重用
            self.cached_child_sampling_params = child_sampling_params
        else:
            # 每个子请求获得一个带有唯一 seed 的克隆
            child_sampling_params.seed = seed + index
        return child_sampling_params

    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
        """获取子请求 ID 和采样参数。

        Args:
            index: 在 n 个子请求中的索引

        Returns:
            (请求 ID, sampling_params) 元组
        """
        child_req_id = f"{index}_{self.request_id}"
        self.child_requests.add(child_req_id)
        return child_req_id, self._get_child_sampling_params(index)

    @property
    def n(self) -> int:
        """返回需要生成的候选输出数量。"""
        return self.sampling_params.n

    def get_outputs(
        self,
        child_request_id: str,
        completion_output: CompletionOutput,
    ) -> tuple[list[CompletionOutput], bool]:
        """处理子请求输出并返回聚合结果。

        Args:
            child_request_id: 子请求 ID
            completion_output: 完成输出

        Returns:
            (输出列表，是否全部完成) 元组
        """
        already_finished_and_returned: bool = False
        if completion_output.finished():
            if child_request_id in self.child_requests:
                self.child_requests.remove(child_request_id)
            else:
                # 子请求 ID 不在 child_requests 中，
                # 表示该请求已在之前的批处理步骤中完成并已返回给客户端
                already_finished_and_returned = True

        if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:
            # 如果是流式模式，只返回当前输出
            # 不要再次向客户端输出已完成并返回的子请求
            outputs = [] if already_finished_and_returned else [completion_output]
        else:
            # 如果不是流式模式，聚合 n 个最终输出
            self.output_aggregator[completion_output.index] = completion_output
            outputs = [] if self.child_requests else self.output_aggregator

        finished = not self.child_requests
        return outputs, finished

    def observe_num_generation_tokens(self, num_generation_tokens: int):
        """更新最大生成 token 数。

        Args:
            num_generation_tokens: 当前生成 token 数

        Returns:
            更新后的最大生成 token 数
        """
        self.max_num_generation_tokens = max(
            num_generation_tokens, self.max_num_generation_tokens
        )
        return self.max_num_generation_tokens

    @staticmethod
    def observe_finished_request(
        parent_req: "ParentRequest | None",
        iteration_stats: IterationStats,
        num_generation_tokens: int,
    ):
        """观察已完成的请求并更新迭代统计信息。

        这是静态方法，用于在请求完成时记录统计信息。

        Args:
            parent_req: 父请求（如果存在）
            iteration_stats: 迭代统计信息
            num_generation_tokens: 生成 token 数
        """
        n_param = parent_req.n if parent_req is not None else 1

        if parent_req is not None:
            num_generation_tokens = parent_req.observe_num_generation_tokens(
                num_generation_tokens
            )

        # 子请求已完成，现在可以记录到迭代统计信息
        if parent_req is None or not parent_req.child_requests:
            iteration_stats.max_num_generation_tokens_iter.append(num_generation_tokens)
            iteration_stats.n_params_iter.append(n_param)
