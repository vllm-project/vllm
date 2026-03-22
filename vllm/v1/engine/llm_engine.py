# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""传统 LLMEngine 包装器模块。

本模块实现了 vLLM V1 引擎的传统包装器，用于向后兼容：
- 提供与 V0 引擎兼容的接口
- 处理输入输出处理
- 管理请求生命周期
- 支持 LoRA 适配器管理
- 支持性能分析、睡眠/唤醒等功能
- 统计数据记录

主要类：
- LLMEngine: 传统引擎包装器类
"""
from collections.abc import Callable, Mapping
from copy import copy
from typing import Any

import torch.nn as nn
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import ProcessorInputs, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class LLMEngine:
    """传统 LLMEngine 包装器，用于向后兼容。

    提供与 V0 引擎兼容的接口，支持：
    - 同步请求处理
    - 流式输出
    - 并行采样（n > 1）
    - LoRA 适配器管理
    - 性能分析
    - 睡眠/唤醒控制
    - 统计数据记录
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        """初始化 LLMEngine。

        Args:
            vllm_config: 全局配置
            executor_class: 执行器类
            log_stats: 是否记录统计数据
            aggregate_engine_logging: 是否聚合引擎日志
            usage_context: 使用上下文
            stat_loggers: 统计日志器列表
            mm_registry: 多模态注册表
            use_cached_outputs: 是否使用缓存输出（已废弃）
            multiprocess_mode: 是否启用多进程模式
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config

        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

        self.log_stats = log_stats

        parallel_config = vllm_config.parallel_config
        executor_backend = parallel_config.distributed_executor_backend

        self.external_launcher_dp = (
            parallel_config.data_parallel_size > 1
            and executor_backend == "external_launcher"
        )
        # important: init dp group before init the engine_core
        # In the decoupled engine case this is handled in EngineCoreProc.
        if (
            not multiprocess_mode
            and parallel_config.data_parallel_size > 1
            and not self.external_launcher_dp
        ):
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None
        self.should_execute_dummy_batch = False

        self.renderer = renderer = renderer_from_config(self.vllm_config)
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.renderer,
            self.model_config.io_processor_plugin,
        )

        # Convert TokPrompt --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        if not multiprocess_mode:
            # for v0 compatibility
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

        if self.external_launcher_dp:
            # If we use DP in external launcher mode, we reuse the
            # existing DP group used for data communication.
            self.dp_group = get_dp_group().cpu_group

        # Don't keep the dummy data in memory
        self.reset_mm_cache()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        """从 VllmConfig 创建 LLMEngine 实例。

        Args:
            vllm_config: 全局配置
            usage_context: 使用上下文
            stat_loggers: 统计日志器列表
            disable_log_stats: 是否禁用统计日志

        Returns:
            LLMEngine 实例
        """
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """从引擎参数创建 LLMEngine 实例。

        Args:
            engine_args: 引擎参数
            usage_context: 使用上下文
            stat_loggers: 统计日志器列表
            enable_multiprocessing: 是否启用多进程

        Returns:
            LLMEngine 实例
        """

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    def get_num_unfinished_requests(self) -> int:
        """获取未完成的请求数量。

        Returns:
            未完成请求数
        """
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        """检查是否有未完成的请求。

        Returns:
            是否有未完成请求
        """
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
            return has_unfinished or self.engine_core.dp_engines_running()
        return self.has_unfinished_requests_dp(has_unfinished)

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        """在数据并行场景下检查是否有未完成的请求。

        Args:
            has_unfinished: 当前引擎的未完成请求状态

        Returns:
            所有 DP 引擎的聚合状态
        """
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished
        )
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """获取支持的任务类型列表。

        Returns:
            支持的任务类型元组
        """
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = self.engine_core.get_supported_tasks()

        return self._supported_tasks

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """中止请求，从 EngineCore 和 Detokenizer 中移除。

        Args:
            request_ids: 请求 ID 列表
            internal: 是否为内部调用
        """

        request_ids = self.output_processor.abort_requests(request_ids, internal)
        self.engine_core.abort_requests(request_ids)

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType | ProcessorInputs,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> str:
        """添加请求到引擎进行处理。

        负责：
        - 输入验证
        - 处理原始输入为 EngineCoreRequest
        - 支持并行采样（n > 1）
        - 添加请求到输出处理器和引擎核心

        Args:
            request_id: 请求 ID
            prompt: 提示词或处理器输入
            params: 采样参数或池化参数
            arrival_time: 到达时间
            lora_request: LoRA 请求
            tokenization_kwargs: 分词参数字典
            trace_headers: 分布式追踪头信息
            priority: 请求优先级
            prompt_text: 提示词文本

        Returns:
            请求 ID

        Raises:
            TypeError: request_id 不是字符串时抛出
        """
        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to LLMEngine.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "LLMEngine.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        self.input_processor.assign_request_id(request)

        req_id = request.request_id

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_text, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return req_id

        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request)
        for idx in range(n):
            request_id, child_params = parent_req.get_child_info(idx)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params

            # Make a new RequestState and queue.
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)

        return req_id

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        """执行引擎调度步骤。

        负责：
        1. 从 EngineCore 获取 EngineCoreOutput
        2. 处理输出生成 RequestOutput
        3. 中止因停止字符串完成的请求
        4. 记录统计数据

        Returns:
            请求输出列表
        """
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        with record_function_or_nullcontext("llm_engine step: get_output"):
            outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            iteration_stats = IterationStats() if self.log_stats else None
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats,
            )
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

        # 3) Abort any reqs that finished due to stop strings.
        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        with record_function_or_nullcontext("llm_engine step: record_stats"):
            if (
                self.logger_manager is not None
                and outputs.scheduler_stats is not None
                and len(outputs.outputs) > 0
            ):
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.renderer.stat_mm_cache(),
                )
                self.do_log_stats_with_interval()

        return processed_outputs.request_outputs

    def start_profile(self, profile_prefix: str | None = None):
        """启动性能分析。

        Args:
            profile_prefix: 性能分析文件前缀
        """
        self.engine_core.profile(True, profile_prefix)

    def stop_profile(self):
        """停止性能分析。"""
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        """重置多模态缓存。"""
        self.renderer.clear_mm_cache()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """重置前缀缓存。

        Args:
            reset_running_requests: 是否重置运行中的请求
            reset_connector: 是否重置连接器

        Returns:
            是否成功重置
        """
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """重置编码器缓存以使所有缓存的编码器输出失效。

        当模型权重更新时应调用此方法，以确保不会重复使用
        旧权重计算的过时视觉嵌入。
        """
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        """使引擎进入睡眠状态。

        Args:
            level: 睡眠级别
            mode: 暂停模式
        """
        self.engine_core.sleep(level, mode)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    def wake_up(self, tags: list[str] | None = None):
        """唤醒引擎。

        Args:
            tags: 唤醒标签
        """
        self.engine_core.wake_up(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    def is_sleeping(self) -> bool:
        """检查引擎是否处于睡眠状态。

        Returns:
            是否处于睡眠状态
        """
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        """获取性能指标快照。

        Returns:
            指标列表

        Raises:
            AssertionError: 统计日志未启用时抛出
        """
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        """返回分词器。

        Returns:
            分词器实例或 None
        """
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        """获取分词器。

        Returns:
            分词器实例
        """
        return self.renderer.get_tokenizer()

    def do_log_stats(self) -> None:
        """记录统计数据（如果日志记录已启用）。"""
        if self.logger_manager:
            self.logger_manager.log()

    def do_log_stats_with_interval(self) -> None:
        """当时间间隔过去后记录统计数据。"""
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = now
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            self.do_log_stats()
            self._last_log_time = now

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """加载新的 LoRA 适配器到引擎中供未来请求使用。

        Args:
            lora_request: LoRA 请求

        Returns:
            是否成功添加
        """
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """移除已加载的 LoRA 适配器。

        Args:
            lora_id: LoRA ID

        Returns:
            是否成功移除
        """
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """列出所有已注册的适配器 ID。

        Returns:
            适配器 ID 集合
        """
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """防止适配器被驱逐。

        Args:
            lora_id: LoRA ID

        Returns:
            是否成功固定
        """
        return self.engine_core.pin_lora(lora_id)

    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        """在 Worker 上执行集体 RPC 调用。

        Args:
            method: 方法名或可调用对象
            timeout: 超时时间（秒）
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            RPC 结果列表
        """
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """在模型上应用函数。

        Args:
            func: 应用于模型的可调用对象

        Returns:
            函数结果列表
        """
        return self.collective_rpc("apply_model", args=(func,))

    def __del__(self):
        """析构函数，清理数据并行组资源。"""
        dp_group = getattr(self, "dp_group", None)
        if dp_group is not None and not self.external_launcher_dp:
            stateless_destroy_torch_distributed_process_group(dp_group)
