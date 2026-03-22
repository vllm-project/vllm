# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM V1 异步 LLM 引擎包装器模块。

本模块实现了 vLLM V1 引擎的异步接口，提供以下核心功能：
- 请求的添加和处理（add_request, generate）
- 输出处理循环（output_handler）
- 引擎控制功能（pause_generation, resume_generation, abort）
- 支持流式输入、池化任务、LoRA 等功能
- 弹性 EP 扩缩容支持
- 分布式追踪和统计日志
"""
import asyncio
import os
import socket
import time
import warnings
from collections.abc import AsyncGenerator, Iterable, Mapping
from copy import copy
from typing import Any

import torch

import vllm.envs as envs
from vllm import TokensPrompt
from vllm.config import VllmConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.entrypoints.serve.elastic_ep.middleware import set_scaling_elastic_ep
from vllm.inputs import ProcessorInputs, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import STREAM_FINISHED, PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.usage.usage_lib import UsageContext
from vllm.utils.async_utils import cancel_task_threadsafe
from vllm.utils.collection_utils import as_list
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,
    StatLoggerManager,
    load_stat_logger_plugin_factories,
)
from vllm.v1.metrics.prometheus import shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)


class InputStreamError(Exception):
    """输入流生成器错误的包装类。

    用于将用户输入生成器的错误传播，而不会将其包装在 EngineGenerateError 中。

    Attributes:
        cause: 原始异常
    """

    def __init__(self, cause: Exception):
        self.cause = cause
        super().__init__(str(cause))


class AsyncLLM(EngineClient):
    """vLLM 引擎的异步包装器。

    提供完整的异步生成接口，支持：
    - 单个和批量请求处理
    - 流式输入
    - 并行采样（n > 1）
    - 池化任务
    - LoRA 适配器管理
    - 引擎控制（暂停/恢复/中止）
    - 弹性 EP 扩缩容
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        """初始化 AsyncLLM。

        Args:
            vllm_config: 全局配置
            executor_class: 执行器类（如 MultiprocExecutor）
            log_stats: 是否记录统计信息
            usage_context: 使用上下文
            mm_registry: 多模态注册表
            use_cached_outputs: 是否使用缓存输出
            log_requests: 是否记录请求日志
            start_engine_loop: 是否启动引擎循环
            stat_loggers: 自定义统计日志记录器列表
            aggregate_engine_logging: 是否聚合引擎日志记录
            client_addresses: 客户端地址字典（用于多客户端场景）
            client_count: 客户端数量
            client_index: 当前客户端索引
        """
        # 确保可以序列化自定义的 transformer 配置
        maybe_register_config_serialize_by_value()

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config

        # 初始化分布式追踪
        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

        self.log_requests = log_requests

        # 配置统计日志记录器
        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())

        has_custom_loggers = bool(custom_stat_loggers)
        self.log_stats = log_stats or has_custom_loggers
        if not log_stats and has_custom_loggers:
            logger.info(
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )

        # 初始化渲染器和 IO 处理器
        self.renderer = renderer = renderer_from_config(self.vllm_config)
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.renderer,
            self.model_config.io_processor_plugin,
        )

        # 输入处理器：将 TokPrompt 转换为 EngineCoreRequest
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # 输出处理器：将 EngineCoreOutputs 转换为 RequestOutput
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # 引擎核心客户端（在后台进程中启动引擎）
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # 日志记录器管理器
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=custom_stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        self._client_count = client_count

        # 输出处理任务
        self.output_handler: asyncio.Task | None = None
        try:
            # 如果在 asyncio 事件循环中，立即启动输出处理器
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

        # 初始化性能分析器（如果启用）
        if (
            vllm_config.profiler_config.profiler == "torch"
            and not vllm_config.profiler_config.ignore_frontend
        ):
            profiler_dir = vllm_config.profiler_config.torch_profiler_dir
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                profiler_dir,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=vllm_config.profiler_config.torch_profiler_with_stack,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profiler_dir,
                    worker_name=worker_name,
                    use_gzip=vllm_config.profiler_config.torch_profiler_use_gzip,
                ),
            )
        else:
            self.profiler = None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncLLM":
        """从 VllmConfig 创建 AsyncLLM 实例。

        Args:
            vllm_config: 全局配置
            start_engine_loop: 是否启动引擎循环
            usage_context: 使用上下文
            stat_loggers: 自定义统计日志记录器
            enable_log_requests: 是否启用请求日志
            aggregate_engine_logging: 是否聚合引擎日志记录
            disable_log_stats: 是否禁用统计日志
            client_addresses: 客户端地址字典
            client_count: 客户端数量
            client_index: 当前客户端索引

        Returns:
            AsyncLLM 实例
        """
        # 创建 LLMEngine
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """从 EngineArgs 创建 AsyncLLM 实例。

        Args:
            engine_args: 引擎参数
            start_engine_loop: 是否启动引擎循环
            usage_context: 使用上下文
            stat_loggers: 自定义统计日志记录器

        Returns:
            AsyncLLM 实例
        """
        # 创建引擎配置
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # 创建 AsyncLLM
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def __del__(self):
        """析构函数，确保关闭时清理资源。"""
        self.shutdown()

    def shutdown(self, timeout: float | None = None) -> None:
        """关闭引擎，清理后台进程和 IPC。

        Args:
            timeout: 关闭超时时间（秒）
        """
        shutdown_prometheus()

        if renderer := getattr(self, "renderer", None):
            renderer.shutdown()

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown(timeout=timeout)

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            cancel_task_threadsafe(handler)

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """获取支持的任务类型列表。

        Returns:
            支持的任务类型元组
        """
        if not hasattr(self, "_supported_tasks"):
            # 缓存结果
            self._supported_tasks = await self.engine_core.get_supported_tasks_async()

        return self._supported_tasks

    async def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest
        | PromptType
        | ProcessorInputs
        | AsyncGenerator[StreamingInput, None],
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
        reasoning_ended: bool | None = None,
    ) -> RequestOutputCollector:
        """添加新请求到 AsyncLLM。

        Args:
            request_id: 请求 ID
            prompt: 输入提示词（可以是 EngineCoreRequest、PromptType、ProcessorInputs 或流式输入生成器）
            params: 采样参数或池化参数
            arrival_time: 到达时间戳
            lora_request: LoRA 适配器请求
            tokenization_kwargs: 分词参数字典
            trace_headers: 分布式追踪头
            priority: 请求优先级
            data_parallel_rank: 数据并行秩
            prompt_text: 提示词文本
            reasoning_ended: 推理是否已结束

        Returns:
            请求输出收集器
        """

        if self.errored:
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)

        # 检查 KV 共享快速预填充与 prompt logprobs 的兼容性
        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and not is_pooling
            and params.prompt_logprobs
        ):
            raise ValueError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        # 处理流式输入
        if isinstance(prompt, AsyncGenerator):
            if reasoning_ended is not None:
                raise NotImplementedError

            # 流式输入场景
            return await self._add_streaming_input_request(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
                data_parallel_rank,
            )

        # 将输入转换为请求
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to AsyncLLM.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            # 使用输入处理器处理输入
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=await self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
            )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        if reasoning_ended is not None:
            request.reasoning_ended = reasoning_ended

        # 分配请求 ID
        self.input_processor.assign_request_id(request)

        # 在首次调用 add_request() 时启动 output_handler
        # 这样可以在事件循环之前调用 __init__，从而在 OpenAI 服务器中优雅地处理启动失败
        self._run_output_handler()

        # 为请求创建新的输出收集器
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # 使用在 process_inputs() 中可能已更新的克隆参数
        params = request.params

        # 处理单个请求或池化请求
        if is_pooling or params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        # 并行采样场景（n > 1）：分发子请求
        parent_params = params
        assert isinstance(parent_params, SamplingParams)

        # 为 n > 1 创建父请求来管理子请求
        parent_request = ParentRequest(request)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            )
        return queue

    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        """添加单个请求到输出处理器和引擎核心。

        Args:
            request: 引擎核心请求
            prompt: 提示词文本
            parent_req: 父请求（并行采样场景）
            index: 子请求索引
            queue: 输出收集器
        """
        # 将请求添加到输出处理器（当前进程）
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # 将 EngineCoreRequest 添加到引擎核心（独立进程）
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    async def _add_streaming_input_request(
        self,
        request_id: str,
        input_stream: AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> RequestOutputCollector:
        """添加流式输入请求。

        Args:
            request_id: 请求 ID
            input_stream: 输入流生成器
            sampling_params: 采样参数
            arrival_time: 到达时间
            lora_request: LoRA 请求
            tokenization_kwargs: 分词参数
            trace_headers: 追踪头
            priority: 优先级
            data_parallel_rank: 数据并行秩

        Returns:
            请求输出收集器
        """
        self._validate_streaming_input_sampling_params(sampling_params)

        inputs = dict(
            supported_tasks=await self.get_supported_tasks(),
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )

        # 避免重复克隆采样参数
        if not sampling_params.skip_clone:
            sampling_params = sampling_params.clone()
            sampling_params.skip_clone = True

        # 创建用于验证的请求，也在输入流关闭时用作完成信号
        final_req = self.input_processor.process_inputs(
            request_id=request_id,
            prompt=TokensPrompt(prompt_token_ids=[0]),
            params=sampling_params,
            **inputs,  # type: ignore[arg-type]
        )
        self.input_processor.assign_request_id(final_req)
        internal_req_id = final_req.request_id

        queue = RequestOutputCollector(sampling_params.output_kind, internal_req_id)

        async def handle_inputs():
            cancelled = False
            try:
                async for input_chunk in input_stream:
                    sp = input_chunk.sampling_params
                    if sp:
                        self._validate_streaming_input_sampling_params(sp)
                    else:
                        sp = sampling_params
                    # TODO(nick): 避免重新验证重复使用的采样参数
                    req = self.input_processor.process_inputs(
                        request_id=internal_req_id,
                        prompt=input_chunk.prompt,
                        params=sp,
                        resumable=True,
                        **inputs,  # type: ignore[arg-type]
                    )
                    req.external_req_id = request_id
                    if req.prompt_embeds is not None:
                        raise ValueError(
                            "prompt_embeds not supported for streaming inputs"
                        )
                    prompt_text, _, _ = extract_prompt_components(
                        self.model_config, input_chunk.prompt
                    )
                    await self._add_request(req, prompt_text, None, 0, queue)
            except (asyncio.CancelledError, GeneratorExit):
                cancelled = True
            except Exception as error:
                # 包装在 InputStreamError 中，以便 generate() 可以传播它
                # 而不会包装在 EngineGenerateError 中
                queue.put(InputStreamError(error))
            finally:
                queue._input_stream_task = None
                if not cancelled:
                    # 发送空最终请求以表示输入已完成
                    # 如果已取消（会话被中止）则不发送
                    await self._add_request(final_req, None, None, 0, queue)

        # 确保输出处理器正在运行
        self._run_output_handler()

        queue._input_stream_task = asyncio.create_task(handle_inputs())
        return queue

    @staticmethod
    def _validate_streaming_input_sampling_params(
        params: SamplingParams | PoolingParams,
    ):
        """验证流式输入采样参数。

        流式输入不支持以下场景：
        - 池化模型
        - n > 1（并行采样）
        - RequestOutputKind.FINAL_ONLY 模式
        - 自定义停止字符串

        Args:
            params: 采样参数或池化参数

        Raises:
            ValueError: 参数不支持流式输入时抛出
        """
        if (
            not isinstance(params, SamplingParams)
            or params.n > 1
            or params.output_kind == RequestOutputKind.FINAL_ONLY
            or params.stop
        ):
            raise ValueError(
                "Input streaming not currently supported "
                "for pooling models, n > 1, request_kind = FINAL_ONLY "
                "or with stop strings."
            )

    # TODO: 我们应该支持在一次调用中处理多个提示词，就像 LLM.generate 一样。
    # 这样对于多提示词完成请求，我们不需要向核心进程发送多条消息，
    # 也不需要多个流，这些流无论如何都会在 API 服务器中重新多路复用。
    async def generate(
        self,
        prompt: EngineCoreRequest
        | PromptType
        | ProcessorInputs
        | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """生成请求的主函数，由 API 服务器调用。

        处理流程：
        1. 创建与请求对应的 AsyncStream
        2. 处理输入
        3. 将请求添加到 Detokenizer
        4. 将请求添加到 EngineCore（独立进程）

        一个单独的 output_handler 循环在后台 AsyncIO 任务中运行，
        从 EngineCore 拉取输出并将其放入每个请求的 AsyncStream 中。

        generate() 的调用者迭代返回的 AsyncGenerator，
        将 RequestOutput 返回给调用者。

        Args:
            prompt: 输入提示词
            sampling_params: 采样参数
            request_id: 请求 ID
            prompt_text: 提示词文本
            lora_request: LoRA 请求
            tokenization_kwargs: 分词参数
            trace_headers: 追踪头
            priority: 优先级
            data_parallel_rank: 数据并行秩
            reasoning_ended: 推理是否已结束

        Yields:
            RequestOutput 对象

        Raises:
            asyncio.CancelledError: 请求被取消时
            GeneratorExit: 生成器退出时
            EngineDeadError: 引擎死亡时
            ValueError: 请求验证错误
            InputStreamError: 输入流错误
            EngineGenerateError: 其他意外错误
        """

        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                prompt_text=prompt_text,
                reasoning_ended=reasoning_ended,
            )

            # output_handler 任务将项目推入队列
            # 此任务从队列中取出并返回给调用者
            finished = False
            while not finished:
                # 注意：尽可能非等待地排空队列（避免负载下的任务切换，有助于提高性能）
                out = q.get_nowait() or await q.get()

                # 注意：OutputProcessor 和 EngineCore 都根据 finished 自行处理请求清理
                assert isinstance(out, RequestOutput)
                finished = out.finished
                if out is not STREAM_FINISHED:
                    yield out

        # 如果客户端断开连接，generate() 会被取消或生成器被垃圾回收
        # 所以我们在这里中止请求
        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # 引擎已死亡。不要中止，因为我们已经关闭。
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # 请求验证错误
        except ValueError as e:
            if self.log_requests:
                logger.info("Request %s failed (bad request): %s.", request_id, e)
            raise

        # 来自输入流生成器的错误 - 直接传播
        except InputStreamError as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed (input error): %s.", request_id, e)
            raise e.cause from e

        # generate() 任务中的意外错误（可能是可恢复的）
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                try:
                    s = f"{e.__class__.__name__}: {e}"
                except Exception as e2:
                    s = (
                        f"{e.__class__.__name__}: "
                        "error during printing an exception of class"
                        + e2.__class__.__name__
                    )
                logger.info("Request %s failed due to %s.", request_id, s)
            raise EngineGenerateError() from e
        finally:
            if q is not None:
                q.close()

    def _run_output_handler(self):
        """后台循环：从 EngineCore 拉取数据并推送到 AsyncStream。

        该处理循环在后台任务中运行，负责：
        1. 从 EngineCore 获取输出
        2. 使用 OutputProcessor 处理输出
        3. 因停止字符串而完成的请求进行中止
        4. 记录统计日志
        """

        if self.output_handler is not None:
            return

        # 确保任务没有对 AsyncLLM 对象的循环引用
        # 否则它无法被正确垃圾回收和清理
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        # 我们使用可变列表来存储 logger_manager，以便在弹性 EP 扩展期间
        # （见 scale_elastic_ep）无需通过 self 创建循环引用即可更新
        self._logger_ref = [self.logger_manager]
        logger_ref = self._logger_ref
        renderer = self.renderer
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

        async def output_handler():
            try:
                while True:
                    # 1. 从 EngineCore 拉取 EngineCoreOutputs
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    iteration_stats = (
                        IterationStats() if (log_stats and num_outputs) else None
                    )

                    # 将输出分割成最多 VLLM_V1_OUTPUT_PROC_CHUNK_SIZE 的块
                    # 以免阻塞事件循环太久
                    engine_core_outputs = outputs.outputs
                    for start in range(0, num_outputs, chunk_size):
                        end = start + chunk_size
                        outputs_slice = engine_core_outputs[start:end]
                        # 2. 处理 EngineCoreOutputs
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        # 注意：RequestOutputs 已被推送到各自的队列
                        assert not processed_outputs.request_outputs

                        # 在块之间允许其他 asyncio 任务运行
                        if end < num_outputs:
                            await asyncio.sleep(0)

                        # 3. 中止因停止字符串而完成的请求
                        if processed_outputs.reqs_to_abort:
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort
                            )

                    output_processor.update_scheduler_stats(outputs.scheduler_stats)

                    # 4. 日志记录
                    # TODO(rob): 一旦 Prometheus 开销不可忽略，就将其放入协程并在后台线程中启动
                    if logger_ref[0]:
                        logger_ref[0].record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                            mm_cache_stats=renderer.stat_mm_cache(),
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        """在 OutputProcessor 和 EngineCore 中中止请求。

        Args:
            request_id: 请求 ID 或 ID 列表
            internal: 是否为内部中止（不向用户报告）
        """
        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        await self.engine_core.abort_requests_async(all_request_ids)

        if self.log_requests:
            logger.info("Aborted request(s) %s.", ",".join(request_ids))

    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool | None = None,
        clear_cache: bool = True,
    ) -> None:
        """暂停生成以允许模型权重更新。

        所有模式处理（中止/等待/保持）和缓存清除都在引擎中完成。
        在调用恢复之前，不会调度新的生成/编码请求。

        Args:
            mode: 如何处理进行中的请求：
                - "abort": 立即中止所有进行中的请求（默认）
                - "wait": 等待进行中的请求完成
                - "keep": 冻结队列中的请求，它们在 resume_generation 时恢复
            wait_for_inflight_requests: 已弃用：使用 mode 参数
            clear_cache: 排空后是否清除 KV 缓存和前缀缓存
        """
        if wait_for_inflight_requests:
            warnings.warn(
                "The `wait_for_inflight_requests` parameter in "
                "`AsyncLLM.pause_generation()` is deprecated. "
                "Please use `mode` argument instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "wait"
        await self.engine_core.pause_scheduler_async(mode=mode, clear_cache=clear_cache)
        # 短暂睡眠以帮助确保在方法返回之前返回任何进行中请求的最终输出
        # 这些输出在等待空闲完成事件之前从引擎输出，但涉及输出处理中的其他异步任务
        # 注意：这对于正确性不是必需的，只是为了从调用者角度来看事件排序更直观
        await asyncio.sleep(0.02)

    async def resume_generation(self) -> None:
        """在 pause_generation 之后恢复生成。"""
        await self.engine_core.resume_scheduler_async()

    async def is_paused(self) -> bool:
        """返回引擎当前是否已暂停。"""
        return await self.engine_core.is_scheduler_paused_async()

    async def encode(
        self,
        prompt: PromptType | ProcessorInputs,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """编码请求的主函数，由 API 服务器调用。

        处理流程：
        1. 创建与请求对应的 AsyncStream
        2. 处理输入
        3. 将请求添加到 EngineCore（独立进程）

        一个单独的 output_handler 循环在后台 AsyncIO 任务中运行，
        从 EngineCore 拉取输出并将其放入每个请求的 AsyncStream 中。

        encode() 的调用者迭代返回的 AsyncGenerator，
        将 PoolingRequestOutput 返回给调用者。

        Args:
            prompt: 输入提示词
            pooling_params: 池化参数
            request_id: 请求 ID
            lora_request: LoRA 请求
            trace_headers: 追踪头
            priority: 优先级
            tokenization_kwargs: 分词参数
            reasoning_ended: 推理是否已结束

        Yields:
            PoolingRequestOutput 对象

        Raises:
            asyncio.CancelledError: 请求被取消时
            EngineDeadError: 引擎死亡时
            ValueError: 请求验证错误
            EngineGenerateError: 其他意外错误
        """

        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                reasoning_ended=reasoning_ended,
            )

            # output_handler 任务将项目推入队列
            # 此任务从队列中取出并返回给调用者
            finished = False
            while not finished:
                # 注意：尽可能非等待地排空队列（避免负载下的任务切换，有助于提高性能）
                out = q.get_nowait() or await q.get()
                assert isinstance(out, PoolingRequestOutput)
                # 注意：OutputProcessor 和 EngineCore 都根据 finished 自行处理请求清理
                finished = out.finished
                yield out

        # 如果客户端断开连接，generate() 会被取消
        # 所以我们在这里中止请求
        except asyncio.CancelledError:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # 引擎已死亡。不要中止，因为我们已经关闭。
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # 请求验证错误
        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # encode() 任务中的意外错误（可能是可恢复的）
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e
        finally:
            if q is not None:
                q.close()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        """返回分词器。"""
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        """获取分词器。"""
        return self.renderer.get_tokenizer()

    async def is_tracing_enabled(self) -> bool:
        """返回是否启用了分布式追踪。"""
        return self.observability_config.otlp_traces_endpoint is not None

    async def do_log_stats(self) -> None:
        """执行统计日志记录。"""
        if self.logger_manager:
            self.logger_manager.log()

    async def check_health(self) -> None:
        """检查引擎健康状态。"""
        logger.debug("Called check_health.")
        if self.errored:
            raise self.dead_error

    async def start_profile(self, profile_prefix: str | None = None) -> None:
        """启动性能分析。

        Args:
            profile_prefix: 性能分析文件前缀
        """
        coros = [self.engine_core.profile_async(True, profile_prefix)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.start))
        await asyncio.gather(*coros)

    async def stop_profile(self) -> None:
        """停止性能分析。"""
        coros = [self.engine_core.profile_async(False)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.stop))
        await asyncio.gather(*coros)

    async def reset_mm_cache(self) -> None:
        """重置多模态缓存。"""
        self.renderer.clear_mm_cache()
        await self.engine_core.reset_mm_cache_async()

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """重置前缀缓存。

        Args:
            reset_running_requests: 是否重置进行中的请求
            reset_connector: 是否重置连接器

        Returns:
            是否成功重置
        """
        return await self.engine_core.reset_prefix_cache_async(
            reset_running_requests, reset_connector
        )

    async def reset_encoder_cache(self) -> None:
        """重置编码器缓存。"""
        await self.engine_core.reset_encoder_cache_async()

    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        """使引擎进入睡眠状态。

        Args:
            level: 睡眠级别
            mode: 暂停模式
        """
        await self.engine_core.sleep_async(level, mode)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        """唤醒引擎。

        Args:
            tags: 唤醒标签列表
        """
        await self.engine_core.wake_up_async(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    async def is_sleeping(self) -> bool:
        """返回引擎是否正在睡眠。"""
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """加载新的 LoRA 适配器到引擎中供未来请求使用。

        Args:
            lora_request: LoRA 适配器请求

        Returns:
            是否成功添加
        """
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """移除已加载的 LoRA 适配器。

        Args:
            lora_id: LoRA ID

        Returns:
            是否成功移除
        """
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """列出所有已注册的 LoRA 适配器 ID。

        Returns:
            LoRA ID 集合
        """
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """防止 LoRA 适配器被驱逐。

        Args:
            lora_id: LoRA ID

        Returns:
            是否成功固定
        """
        return await self.engine_core.pin_lora_async(lora_id)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """执行集体 RPC 调用。

        Args:
            method: 方法名
            timeout: 超时时间（秒）
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            RPC 调用结果
        """
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs
        )

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        """等待所有请求排空。

        Args:
            drain_timeout: 超时时间（秒）

        Raises:
            TimeoutError: 超时时抛出
        """
        start_time = time.time()
        while time.time() - start_time < drain_timeout:
            if not self.engine_core.dp_engines_running():
                logger.info("Engines are idle, requests have been drained")
                return

            logger.info("Engines are still running, waiting for requests to drain...")
            await asyncio.sleep(1)  # 等待 1 秒后再次检查

        raise TimeoutError(
            f"Timeout reached after {drain_timeout} seconds "
            "waiting for requests to drain."
        )

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ):
        """通过添加或移除引擎核心来扩展数据并行大小。

        Args:
            new_data_parallel_size: 新的数据并行工作节点数量
            drain_timeout: 等待请求排空的最大时间（秒）
        """
        old_data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if old_data_parallel_size == new_data_parallel_size:
            logger.info(
                "Data parallel size is already %s, skipping scale",
                new_data_parallel_size,
            )
            return

        if envs.VLLM_ELASTIC_EP_DRAIN_REQUESTS:
            logger.info(
                "VLLM_ELASTIC_EP_DRAIN_REQUESTS is set, "
                "waiting for requests to drain before scaling"
            )
            await self.wait_for_requests_to_drain(drain_timeout)

        # 重新创建统计日志记录器
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:
            # TODO(rob): 与 Ray 团队沟通后修复此问题
            # 这会在初始化期间取消注册时重置所有 prometheus 指标
            # 需要在这里更好地理解预期行为
            self.logger_manager = StatLoggerManager(
                vllm_config=self.vllm_config,
                engine_idxs=list(range(new_data_parallel_size)),
                custom_stat_loggers=None,
            )
            # 更新可变引用，使 output_handler 能够获取新的日志记录器
            # 而无需通过 self 创建循环引用
            if hasattr(self, "_logger_ref"):
                self._logger_ref[0] = self.logger_manager
            self.logger_manager.log_engine_initialized()

        set_scaling_elastic_ep(True)
        try:
            await self.engine_core.scale_elastic_ep(new_data_parallel_size)
            self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        finally:
            set_scaling_elastic_ep(False)

    @property
    def is_running(self) -> bool:
        """返回引擎是否正在运行。

        在循环启动之前为 None。
        """
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        """返回引擎是否已停止。"""
        return self.errored

    @property
    def errored(self) -> bool:
        """返回引擎是否出错。"""
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        """返回引擎死亡错误。"""
        return EngineDeadError()

    async def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest
    ) -> None:
        """初始化权重传输用于 RL 训练。

        Args:
            request: 权重传输初始化请求，包含后端特定的信息
        """
        from vllm.distributed.weight_transfer.base import (
            WeightTransferInitRequest,
        )

        if isinstance(request, WeightTransferInitRequest):
            init_info_dict = request.init_info
        else:
            raise TypeError(f"Expected WeightTransferInitRequest, got {type(request)}")

        await self.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
        )

    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:
        """批量权重更新用于 RL 训练。

        Args:
            request: 权重更新请求，包含后端特定的更新信息
        """

        if isinstance(request, WeightTransferUpdateRequest):
            update_info_dict = request.update_info
        else:
            raise TypeError(
                f"Expected WeightTransferUpdateRequest, got {type(request)}"
            )

        await self.collective_rpc(
            "update_weights", kwargs={"update_info": update_info_dict}
        )
