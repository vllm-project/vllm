# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""输出处理器模块。

本模块实现了 vLLM V1 引擎的输出处理功能，负责：
- 将 EngineCoreOutput 转换为 RequestOutput
- 增量反词元化
- Logprobs 处理
- 流式输出支持
- 请求状态管理
- 统计数据收集
- 分布式追踪支持

主要类：
- RequestOutputCollector: 请求输出收集器
- OutputProcessorOutput: 输出处理器输出数据类
- StreamingUpdate: 流式输入更新数据类
- RequestState: 请求状态类
- OutputProcessor: 输出处理器
"""
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.outputs import (
    STREAM_FINISHED,
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
)
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import TokenizerLike
from vllm.tracing import (
    SpanAttributes,
    SpanKind,
    extract_trace_context,
    instrument_manual,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import (
    IterationStats,
    LoRARequestStates,
    RequestStateStats,
    SchedulerStats,
)

# shared empty CPU tensor used as a placeholder pooling output
EMPTY_CPU_TENSOR = torch.empty(0, device="cpu")


class RequestOutputCollector:
    """请求输出收集器。

    为每个请求收集流式 RequestOutput，并传递给消费的 asyncio generate 任务。

    当流式传输 deltas 时，如果生产者领先于消费者，RequestOutputs 会被合并。

    Attributes:
        aggregate: 是否聚合输出（DELTA 模式）
        request_id: 请求 ID
        output: 输出或异常
        ready: 就绪事件
        _input_stream_task: 输入流任务
    """

    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        """初始化请求输出收集器。

        Args:
            output_kind: 输出类型
            request_id: 请求 ID
        """

    def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None:
        """非阻塞 put 操作。

        Args:
            output: 请求输出或异常
        """
        if self.output is None or isinstance(output, Exception):
            self.output = output
            self.ready.set()
        elif isinstance(self.output, RequestOutput) and isinstance(
            output, RequestOutput
        ):
            # 这确保具有不同请求索引的请求输出（如果 n > 1）不会相互覆盖
            self.output.add(output, aggregate=self.aggregate)
        elif isinstance(self.output, PoolingRequestOutput) and isinstance(
            output, PoolingRequestOutput
        ):
            self.output = output

    async def get(self) -> RequestOutput | PoolingRequestOutput:
        """Get 操作阻塞直到 put 事件。

        Returns:
            请求输出或池化请求输出

        Raises:
            Exception: 如果存储的是异常则抛出
        """
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None:
        """非阻塞 get 操作。

        Returns:
            请求输出或池化请求输出，如果没有则为 None

        Raises:
            Exception: 如果存储的是异常则抛出
        """
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    def close(self):
        """关闭收集器，取消输入流任务。"""
        if self._input_stream_task is not None:
            self._input_stream_task.cancel()
        self._input_stream_task = None

    def __del__(self):
        """析构函数，取消输入流任务。"""
        if (task := self._input_stream_task) is not None:
            task.get_loop().call_soon_threadsafe(task.cancel)
            self._input_stream_task = None


@dataclass
class OutputProcessorOutput:
    """输出处理器输出数据类。

    Attributes:
        request_outputs: 请求输出列表
        reqs_to_abort: 需要中止的请求 ID 列表
    """
    request_outputs: list[RequestOutput | PoolingRequestOutput]
    reqs_to_abort: list[str]


@dataclass
class StreamingUpdate:
    """输出处理器的流式输入更新数据。

    包含当当前子请求完成时要应用于请求状态的增量提示数据。

    Attributes:
        prompt: 提示词文本
        prompt_token_ids: 提示词 token IDs
        arrival_time: 到达时间
        final: 是否为最终更新
    """

    prompt: str | None
    prompt_token_ids: list[int] | None
    arrival_time: float
    final: bool = False


class RequestState:
    """请求状态类。

    存储和管理单个请求的处理状态，包括：
    - 请求标识和元数据
    - 提示词和 token IDs
    - 反词元化和 logprobs 处理器
    - 流式输出状态
    - 统计数据
    """

    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        parent_req: ParentRequest | None,
        request_index: int,
        lora_request: LoRARequest | None,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        logprobs_processor: LogprobsProcessor | None,
        detokenizer: IncrementalDetokenizer | None,
        max_tokens_param: int | None,
        arrival_time: float,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
        top_p: float | None = None,
        n: int | None = None,
        temperature: float | None = None,
        stream_input: bool = False,
    ):
        """初始化请求状态。

        Args:
            request_id: 内部请求 ID
            external_req_id: 外部请求 ID
            parent_req: 父请求（用于并行采样）
            request_index: 请求索引
            lora_request: LoRA 请求
            output_kind: 输出类型
            prompt: 提示词文本
            prompt_token_ids: 提示词 token IDs
            prompt_embeds: 提示词嵌入
            logprobs_processor: Logprobs 处理器
            detokenizer: 反词元化器
            max_tokens_param: 最大 token 数参数
            arrival_time: 到达时间
            queue: 请求输出收集器
            log_stats: 是否记录统计
            stream_interval: 流式间隔
            top_p: Top-p 采样参数
            n: 并行采样数
            temperature: 温度参数
            stream_input: 是否支持流式输入
        """
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.parent_req = parent_req
        self.request_index = request_index
        self.lora_request = lora_request
        self.lora_name = lora_request.lora_name if lora_request is not None else None
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )
        self.logprobs_processor = logprobs_processor
        self.detokenizer = detokenizer
        self.max_tokens_param = max_tokens_param
        self.top_p = top_p
        self.n = n
        self.temperature = temperature
        self.is_prefilling = True
        self.queue = queue
        self.num_cached_tokens = 0

        self.stats = RequestStateStats(arrival_time=arrival_time) if log_stats else None

        # Stream Interval
        self.stream_interval = stream_interval
        self.sent_tokens_offset = 0  # 已发送 token 的偏移量

        # Streaming input queue
        self.streaming_input = stream_input
        self.input_chunk_queue: deque[StreamingUpdate] | None = (
            deque() if stream_input else None
        )

    def apply_streaming_update(self, update: StreamingUpdate) -> None:
        """应用流式更新到请求状态。

        Args:
            update: 流式更新数据
        """
        # Apply the update to the request state.
        self.streaming_input = not update.final
        # TODO also include relevant output tokens in new prompt here
        #     (match scheduler behavior).
        if update.prompt:
            self.prompt = (
                (self.prompt + update.prompt) if self.prompt else update.prompt
            )
        if self.prompt_token_ids:
            self.prompt_token_ids.extend(update.prompt_token_ids or ())
        else:
            self.prompt_token_ids = update.prompt_token_ids or []
        assert self.prompt_token_ids is not None
        self.prompt_len = len(self.prompt_token_ids)
        if self.stats is not None:
            self.stats.arrival_time = update.arrival_time
        self.is_prefilling = True

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
    ) -> "RequestState":
        """从新请求创建请求状态。

        Args:
            tokenizer: 分词器
            request: 引擎核心请求
            prompt: 提示词文本
            parent_req: 父请求
            request_index: 请求索引
            queue: 请求输出收集器
            log_stats: 是否记录统计
            stream_interval: 流式间隔

        Returns:
            RequestState 实例
        """
        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            max_tokens_param = sampling_params.max_tokens
            top_p = sampling_params.top_p
            n = sampling_params.n
            temperature = sampling_params.temperature
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            top_p = None
            n = None
            temperature = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind

        assert request.external_req_id is not None
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            parent_req=parent_req,
            request_index=request_index,
            lora_request=request.lora_request,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            top_p=top_p,
            n=n,
            temperature=temperature,
            arrival_time=request.arrival_time,
            queue=queue,
            log_stats=log_stats,
            stream_interval=stream_interval,
            stream_input=request.resumable,
        )

    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        routed_experts: np.ndarray | None = None,
    ) -> RequestOutput | PoolingRequestOutput | None:
        """创建请求输出。

        Args:
            new_token_ids: 新的 token IDs
            pooling_output: 池化输出张量
            finish_reason: 完成原因
            stop_reason: 停止原因
            kv_transfer_params: KV 传输参数
            routed_experts: 路由专家信息

        Returns:
            请求输出或池化请求输出，如果不需要输出则返回 None
        """
        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            return None

        if self.stream_interval > 1:
            assert self.detokenizer is not None

            # Send output request only when
            # 1. It has finished, or
            # 2. It is the first token, or
            # 3. It has reached the stream interval number of tokens
            if not (
                finished
                or self.sent_tokens_offset == 0
                or self.detokenizer.num_output_tokens() - self.sent_tokens_offset
                >= self.stream_interval
            ):
                return None

            if self.output_kind == RequestOutputKind.DELTA:
                # Send tokens from the offset in DELTA mode, otherwise all
                # tokens are sent.
                new_token_ids = self.detokenizer.output_token_ids[
                    self.sent_tokens_offset :
                ]
                self.sent_tokens_offset = self.detokenizer.num_output_tokens()

        external_req_id = self.external_req_id

        if pooling_output is not None:
            return self._new_request_output(
                external_req_id,
                [self._new_pooling_output(pooling_output)],
                finished,
            )

        output = self._new_completion_output(
            new_token_ids, finish_reason, stop_reason, routed_experts
        )

        if self.parent_req is None:
            outputs = [output]
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            if not outputs:
                return None
            external_req_id = self.parent_req.external_req_id

        return self._new_request_output(
            external_req_id, outputs, finished, kv_transfer_params
        )

    def _new_request_output(
        self,
        external_req_id: str,
        outputs: list[CompletionOutput] | list[PoolingOutput],
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput:
        """创建新的请求输出。

        Args:
            external_req_id: 外部请求 ID
            outputs: 完成输出或池化输出列表
            finished: 是否已完成
            kv_transfer_params: KV 传输参数

        Returns:
            请求输出或池化请求输出
        """
        # If prompt embeds were used, put placeholder prompt token ids
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)
        assert prompt_token_ids is not None

        first_output = outputs[0]
        if isinstance(first_output, PoolingOutput):
            assert len(outputs) == 1
            return PoolingRequestOutput(
                request_id=external_req_id,
                outputs=first_output,
                num_cached_tokens=self.num_cached_tokens,
                prompt_token_ids=prompt_token_ids,
                finished=finished,
            )
        assert self.logprobs_processor is not None
        if self.output_kind == RequestOutputKind.DELTA:
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
        else:
            prompt_logprobs = self.logprobs_processor.prompt_logprobs

        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            lora_request=self.lora_request,
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=cast(list[CompletionOutput], outputs),
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            num_cached_tokens=self.num_cached_tokens,
            metrics=self.stats,
        )

    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        routed_experts: np.ndarray | None = None,
    ) -> CompletionOutput:
        """创建新的完成输出。

        Args:
            token_ids: token IDs 列表
            finish_reason: 完成原因
            stop_reason: 停止原因
            routed_experts: 路由专家信息

        Returns:
            完成输出
        """
        assert self.detokenizer is not None
        assert self.logprobs_processor is not None
        finished = finish_reason is not None
        delta = self.output_kind == RequestOutputKind.DELTA

        # Prepare text and token_ids, based on delta mode
        text = self.detokenizer.get_next_output_text(finished, delta)
        if not delta:
            token_ids = self.detokenizer.output_token_ids

        # Prepare logprobs, based on delta mode
        logprobs = self.logprobs_processor.logprobs
        if delta and logprobs:
            logprobs = logprobs[-len(token_ids) :]

        return CompletionOutput(
            index=self.request_index,
            text=text,
            token_ids=token_ids,
            routed_experts=routed_experts,
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
        )

    def _new_pooling_output(self, pooling_output: torch.Tensor) -> PoolingOutput:
        """创建新的池化输出。

        Args:
            pooling_output: 池化输出张量

        Returns:
            池化输出
        """
        return PoolingOutput(data=pooling_output)


class OutputProcessor:
    """输出处理器。

    负责将 EngineCoreOutputs 处理为 RequestOutputs，包括：
    - 请求状态管理
    - 增量反词元化
    - Logprobs 处理
    - 流式输出支持
    - 统计数据收集
    - 分布式追踪
    """

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
    ):
        """初始化输出处理器。

        Args:
            tokenizer: 分词器
            log_stats: 是否记录统计
            stream_interval: 流式间隔
            tracing_enabled: 是否启用追踪
        """
        self.log_stats = log_stats
        self.tokenizer = tokenizer
        self.stream_interval = stream_interval
        self.request_states: dict[str, RequestState] = {}
        self.parent_requests: dict[str, ParentRequest] = {}
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
        self.lora_states = LoRARequestStates(log_stats)
        self.tracing_enabled = tracing_enabled

    def get_num_unfinished_requests(self):
        """获取未完成请求数量。

        Returns:
            未完成请求数
        """
        return len(self.request_states)

    def has_unfinished_requests(self) -> bool:
        """检查是否有未完成请求。

        Returns:
            是否有未完成请求
        """
        return len(self.request_states) > 0

    def propagate_error(self, e: Exception):
        """将所有错误传播给所有 generate() 任务。

        Args:
            e: 异常
        """
        for _, state in self.request_states.items():
            assert state.queue is not None
            state.queue.put(e)

    def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
        """中止请求列表。

        request_ids 可以是外部请求 ID（传递给 InputProcessor.process_inputs() 的 ID）
        或内部请求 ID（创建 EngineCoreRequest 时随机生成的 ID）。

        如果提供外部请求 ID，并且该外部请求 ID 用于多个请求，
        则与该外部请求 ID 关联的所有请求都将被中止。

        在并行采样情况下，请求 ID 可用于标识父请求，
        此时关联的子请求也将被中止。

        Args:
            request_ids: 请求 ID 列表
            internal: 是否为内部 ID

        Returns:
            被中止的请求 ID 列表
        """
        internal_req_ids = []
        for request_id in request_ids:
            if internal:
                # Internal ID - this may be a parent request
                internal_req_ids.append(request_id)

                # Remove internal ID from the external->internal mapping
                if req_state := self.request_states.get(request_id):
                    external_req_id = req_state.external_req_id
                    internal_ids = self.external_req_ids[external_req_id]
                    internal_ids.remove(request_id)
                    if not internal_ids:
                        del self.external_req_ids[external_req_id]
            elif internal_ids := self.external_req_ids.pop(request_id, []):
                # External ID - abort all requests in the external->internal mapping
                internal_req_ids.extend(internal_ids)

        request_ids_to_abort = []
        for request_id in internal_req_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                self.lora_states.request_finished(request_id, req_state.lora_name)
                request_ids_to_abort.append(request_id)
                # Produce final abort output.
                if req_state.queue is not None and (
                    request_output := req_state.make_request_output(
                        new_token_ids=[],
                        # Set pooling_output is not None to
                        # correctly enter the abort pooling branch
                        pooling_output=EMPTY_CPU_TENSOR
                        if req_state.detokenizer is None
                        else None,
                        finish_reason=FinishReason.ABORT,
                        stop_reason=None,
                        kv_transfer_params=None,
                    )
                ):
                    req_state.queue.put(request_output)
            elif parent := self.parent_requests.get(request_id):
                # Abort children prior to removing the parent.
                if parent.child_requests:
                    child_reqs = list(parent.child_requests)
                    child_reqs = self.abort_requests(child_reqs, internal=True)
                    request_ids_to_abort.extend(child_reqs)
                self.parent_requests.pop(request_id, None)
        return request_ids_to_abort

    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        """添加请求到处理器。

        Args:
            request: 引擎核心请求
            prompt: 提示词文本
            parent_req: 父请求
            request_index: 请求索引
            queue: 请求输出收集器
        """
        request_id = request.request_id
        req_state = self.request_states.get(request_id)
        if req_state is not None:
            self._update_streaming_request_state(req_state, request, prompt)
            return

        req_state = RequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=self.stream_interval,
        )
        self.request_states[request_id] = req_state
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

        # Track the external_req_id -> [internal_req_id, ...] mapping
        self.external_req_ids[req_state.external_req_id].append(request_id)

    def _update_streaming_request_state(
        self, req_state: RequestState, request: EngineCoreRequest, prompt: str | None
    ) -> None:
        """将流式更新入队而不是立即应用。

        Args:
            req_state: 请求状态
            request: 引擎核心请求
            prompt: 提示词文本
        """
        if not request.resumable:
            # Final request - just mark completion, don't add its dummy tokens.
            if req_state.input_chunk_queue is None:
                # Engine already finished - emit final output and clean up.
                self._finish_request(req_state)
                if req_state.queue is not None:
                    # Emit a final output with finished=True
                    # to unblock the generate() loop.
                    req_state.queue.put(STREAM_FINISHED)
            elif req_state.input_chunk_queue:
                req_state.input_chunk_queue[-1].final = True
            else:
                req_state.streaming_input = False
            return

        update = StreamingUpdate(
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            arrival_time=request.arrival_time,
        )

        # Apply request updates now if the last input already completed.
        if req_state.input_chunk_queue is None:
            req_state.apply_streaming_update(update)
            req_state.input_chunk_queue = deque()
        else:
            # Queue the streaming update otherwise.
            req_state.input_chunk_queue.append(update)

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput:
        """处理 EngineCoreOutputs。

        负责：
        1. 计算统计数据用于日志记录
        2. 反词元化
        3. 创建和处理 RequestOutput 对象：
           - 如果有队列（用于 AsyncLLM），将 RequestOutput 放入队列
             供每个请求的 generate() 任务处理
           - 如果没有队列（用于 LLMEngine），返回 RequestOutput 列表

        注意：
        vLLM V1 最小化对整个批次的 Python 循环次数以确保系统开销最小化。
        这是唯一应该循环 EngineCoreOutputs 的函数。

        Args:
            engine_core_outputs: EngineCoreOutput 列表
            engine_core_timestamp: EngineCore 时间戳
            iteration_stats: 迭代统计

        Returns:
            OutputProcessorOutput 包含处理后的输出
        """

        request_outputs: list[RequestOutput | PoolingRequestOutput] = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # 1) Compute stats for this iteration.
            self._update_stats_from_output(
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )

            new_token_ids = engine_core_output.new_token_ids
            pooling_output = engine_core_output.pooling_output
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params
            routed_experts = engine_core_output.routed_experts
            req_state.num_cached_tokens = engine_core_output.num_cached_tokens
            req_state.is_prefilling = False

            if pooling_output is None:
                assert req_state.detokenizer is not None
                assert req_state.logprobs_processor is not None
                # 2) Detokenize the token ids into text and perform stop checks.
                stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP
                )
                if stop_string:
                    finish_reason = FinishReason.STOP
                    stop_reason = stop_string

                # 3) Compute sample and prompt logprobs for request,
                # if required.
                req_state.logprobs_processor.update_from_output(engine_core_output)

            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(
                new_token_ids,
                pooling_output,
                finish_reason,
                stop_reason,
                kv_transfer_params,
                routed_experts,
            ):
                if req_state.streaming_input:
                    request_output.finished = False

                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

            # Free completed requests.
            if finish_reason is not None:
                if req_state.streaming_input:
                    if req_state.input_chunk_queue:
                        update = req_state.input_chunk_queue.popleft()
                        req_state.apply_streaming_update(update)
                    else:
                        req_state.input_chunk_queue = None
                else:
                    self._finish_request(req_state)
                    if not engine_core_output.finished:
                        # If req not finished in EngineCore, but Detokenizer
                        # detected stop string, abort needed in EngineCore.
                        reqs_to_abort.append(req_id)

                    # Track per-request stats
                    self._update_stats_from_finished(
                        req_state, finish_reason, iteration_stats
                    )
                    if self.tracing_enabled:
                        self.do_tracing(engine_core_output, req_state, iteration_stats)

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    def _finish_request(self, req_state: RequestState) -> None:
        """完成请求并清理状态。

        Args:
            req_state: 请求状态
        """
        req_id = req_state.request_id
        self.request_states.pop(req_id)

        internal_ids = self.external_req_ids[req_state.external_req_id]
        internal_ids.remove(req_id)
        if not internal_ids:
            del self.external_req_ids[req_state.external_req_id]

        # Remove parent request if applicable.
        parent_req = req_state.parent_req
        if parent_req and not parent_req.child_requests:
            self.parent_requests.pop(parent_req.request_id, None)

    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        """更新调度器统计信息。

        Args:
            scheduler_stats: 调度器统计
        """
        self.lora_states.update_scheduler_stats(scheduler_stats)

    def do_tracing(
        self,
        engine_core_output: EngineCoreOutput,
        req_state: RequestState,
        iteration_stats: IterationStats | None,
    ) -> None:
        """执行分布式追踪。

        Args:
            engine_core_output: 引擎核心输出
            req_state: 请求状态
            iteration_stats: 迭代统计
        """
        assert req_state.stats is not None
        assert iteration_stats is not None

        metrics = req_state.stats
        arrival_time_ns = int(metrics.arrival_time * 1e9)
        trace_context = extract_trace_context(engine_core_output.trace_headers)
        prompt_length = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Calculate timing metrics
        e2e_time = iteration_stats.iteration_timestamp - metrics.arrival_time
        queued_time = metrics.scheduled_ts - metrics.queued_ts
        prefill_time = metrics.first_token_ts - metrics.scheduled_ts
        decode_time = metrics.last_token_ts - metrics.first_token_ts
        inference_time = metrics.last_token_ts - metrics.scheduled_ts

        # Build attributes dict
        attributes: dict[str, Any] = {
            SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: (
                metrics.first_token_latency
            ),
            SpanAttributes.GEN_AI_LATENCY_E2E: e2e_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE: queued_time,
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: prompt_length,
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: (
                metrics.num_generation_tokens
            ),
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: prefill_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: decode_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE: inference_time,
            SpanAttributes.GEN_AI_REQUEST_ID: req_state.external_req_id,
        }

        # Add optional request parameters
        if req_state.top_p:
            attributes[SpanAttributes.GEN_AI_REQUEST_TOP_P] = req_state.top_p
        if req_state.max_tokens_param:
            attributes[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = (
                req_state.max_tokens_param
            )
        if req_state.temperature:
            attributes[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = (
                req_state.temperature
            )
        if req_state.n:
            attributes[SpanAttributes.GEN_AI_REQUEST_N] = req_state.n

        instrument_manual(
            span_name="llm_request",
            start_time=arrival_time_ns,
            attributes=attributes,
            context=trace_context,
            kind=SpanKind.SERVER,
        )

    def _update_stats_from_output(
        self,
        req_state: RequestState,
        engine_core_output: EngineCoreOutput,
        engine_core_timestamp: float | None,
        iteration_stats: IterationStats | None,
    ):
        """从输出更新统计信息。

        Args:
            req_state: 请求状态
            engine_core_output: 引擎核心输出
            engine_core_timestamp: EngineCore 时间戳
            iteration_stats: 迭代统计
        """
        if iteration_stats is None:
            return

        assert engine_core_timestamp is not None
        assert req_state.stats is not None
        iteration_stats.update_from_output(
            engine_core_output,
            engine_core_timestamp,
            req_state.is_prefilling,
            req_state.prompt_len,
            req_state.stats,
            self.lora_states,
            req_state.lora_name,
        )

    def _update_stats_from_finished(
        self,
        req_state: RequestState,
        finish_reason: FinishReason | None,
        iteration_stats: IterationStats | None,
    ):
        """从完成的请求更新统计信息。

        Args:
            req_state: 请求状态
            finish_reason: 完成原因
            iteration_stats: 迭代统计
        """
        if iteration_stats is None:
            return

        assert finish_reason is not None
        assert req_state.stats is not None
        iteration_stats.update_from_finished_request(
            finish_reason=finish_reason,
            num_prompt_tokens=req_state.prompt_len,
            max_tokens_param=req_state.max_tokens_param,
            req_stats=req_state.stats,
            num_cached_tokens=req_state.num_cached_tokens,
        )
        self.lora_states.request_finished(req_state.request_id, req_state.lora_name)

        ParentRequest.observe_finished_request(
            req_state.parent_req, iteration_stats, req_state.stats.num_generation_tokens
        )
