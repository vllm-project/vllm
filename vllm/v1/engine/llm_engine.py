# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable, Mapping
from copy import copy
from typing import Any, cast

import torch.nn as nn
from typing_extensions import TypeVar, deprecated

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike, init_tokenizer_from_config
from vllm.tracing import init_tracer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
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
    """Legacy LLMEngine for backwards compatibility."""

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
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        self.log_stats = log_stats

        executor_backend = self.vllm_config.parallel_config.distributed_executor_backend
        parallel_config = vllm_config.parallel_config
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

        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = init_tokenizer_from_config(self.model_config)

        self.input_processor = InputProcessor(self.vllm_config, tokenizer)
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        )

        # OutputProcessor (convert EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(
            self.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
        )
        endpoint = self.observability_config.otlp_traces_endpoint
        if endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", endpoint)
            self.output_processor.tracer = tracer

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

    @property
    @deprecated(
        "`LLMEngine.processor` has been renamed to `LLMEngine.input_processor`. "
        "The old name will be removed in v0.13."
    )
    def processor(self):
        return self.input_processor

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
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
        """Creates an LLM engine from the engine arguments."""

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
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
            return has_unfinished or self.engine_core.dp_engines_running()
        return self.has_unfinished_requests_dp(has_unfinished)

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished
        )
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def abort_request(self, request_ids: list[str]) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        request_ids = self.output_processor.abort_requests(request_ids)
        self.engine_core.abort_requests(request_ids)

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> None:
        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        if isinstance(prompt, EngineCoreRequest):
            request = prompt
        else:
            assert prompt_text is None
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
            )
            if isinstance(prompt, str):
                prompt_text = prompt
            elif isinstance(prompt, Mapping):
                prompt_text = cast(str | None, prompt.get("prompt"))

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_text, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return

        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request_id, params)
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

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
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
            if self.logger_manager is not None and outputs.scheduler_stats is not None:
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.input_processor.stat_mm_cache(),
                )
                self.do_log_stats_with_interval()

        return processed_outputs.request_outputs

    def start_profile(self):
        self.engine_core.profile(True)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        self.input_processor.clear_mm_cache()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(self, reset_running_requests: bool = False) -> bool:
        return self.engine_core.reset_prefix_cache(reset_running_requests)

    def sleep(self, level: int = 1):
        self.engine_core.sleep(level)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    def wake_up(self, tags: list[str] | None = None):
        self.engine_core.wake_up(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.input_processor.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerLike | None) -> None:
        self.input_processor.tokenizer = tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        if self.tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return self.tokenizer

    def do_log_stats(self) -> None:
        """Log stats if logging is enabled."""
        if self.logger_manager:
            self.logger_manager.log()

    def do_log_stats_with_interval(self) -> None:
        """Log stats when the time interval has passed."""
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = now
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            self.do_log_stats()
            self._last_log_time = now

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return self.engine_core.pin_lora(lora_id)

    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.collective_rpc("apply_model", args=(func,))

    def __del__(self):
        if (
            dp_group := getattr(self, "dp_group", None)
            and not self.external_launcher_dp
        ):
            stateless_destroy_torch_distributed_process_group(dp_group)
