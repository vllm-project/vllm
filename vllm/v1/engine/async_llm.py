# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Mapping
from copy import copy
from typing import Optional, Union

import numpy as np

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.envs import VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
from vllm.inputs import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device, cdiv, kill_process_tree
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import (OutputProcessor,
                                             RequestOutputCollector)
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import (LoggingStatLogger, PrometheusStatLogger,
                                     StatLoggerBase)
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

logger = init_logger(__name__)


class AsyncLLM(EngineClient):

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
    ) -> None:
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        assert start_engine_loop

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.log_requests = log_requests
        self.log_stats = log_stats

        # Set up stat loggers; independent set for each DP rank.
        self.stat_loggers: list[list[StatLoggerBase]] = []
        if self.log_stats:
            for i in range(vllm_config.parallel_config.data_parallel_size):
                loggers: list[StatLoggerBase] = []
                if logger.isEnabledFor(logging.INFO):
                    loggers.append(LoggingStatLogger(engine_index=i))
                loggers.append(
                    PrometheusStatLogger(vllm_config, engine_index=i))
                self.stat_loggers.append(loggers)

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)
        self.tokenizer.ping()

        # Processor (converts Inputs --> EngineCoreRequests).
        self.processor = Processor(
            vllm_config=vllm_config,
            tokenizer=self.tokenizer,
            mm_registry=mm_registry,
        )

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(self.tokenizer,
                                                log_stats=self.log_stats)

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=True,
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        self.output_handler: Optional[asyncio.Task] = None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[dict[str, StatLoggerBase]] = None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
    ) -> "AsyncLLM":
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        # FIXME(rob): refactor VllmConfig to include the StatLoggers
        # include StatLogger in the Oracle decision.
        if stat_loggers is not None:
            raise ValueError("Custom StatLoggers are not yet supported on V1. "
                             "Explicitly set VLLM_USE_V1=0 to disable V1.")

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            log_requests=not disable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
        )

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC."""

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()

        if handler := getattr(self, "output_handler", None):
            handler.cancel()

    async def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""

        assert isinstance(params, SamplingParams), \
            "Pooling is not supported in V1"

        # Create a new output collector for the request.
        queue = RequestOutputCollector(output_kind=params.output_kind)

        # Convert Input --> Request.
        request = self.processor.process_inputs(request_id, prompt, params,
                                                arrival_time, lora_request,
                                                trace_headers,
                                                prompt_adapter_request,
                                                priority)

        if params.n == 1:
            await self._add_request(request, None, 0, queue)
            return queue

        # Fan out child requests (for n>1).
        parent_request = ParentRequest(request_id, params)
        for idx in range(params.n):
            request_id, params = parent_request.get_child_info(idx)
            child_request = request if idx == params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = params
            await self._add_request(child_request, parent_request, idx, queue)
        return queue

    async def _add_request(self, request: EngineCoreRequest,
                           parent_req: Optional[ParentRequest], index: int,
                           queue: RequestOutputCollector):

        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        try:
            # We start the output_handler on the first call to generate() so
            # we can call __init__ before the event loop, which enables us
            # to handle startup failure gracefully in the OpenAI server.
            if self.output_handler is None:
                self.output_handler = asyncio.create_task(
                    self._run_output_handler())

            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()

                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                yield out

        # If the request is disconnected by the client, the
        # generate() task will be canceled. So, we abort the
        # request if we end up here.
        except asyncio.CancelledError:
            await self.abort(request_id)
            raise

    async def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        try:
            while True:
                # 1) Pull EngineCoreOutputs from the EngineCore.
                outputs = await self.engine_core.get_output_async()
                num_outputs = len(outputs.outputs)

                iteration_stats = IterationStats() if (
                    self.log_stats and num_outputs) else None

                # Split outputs into chunks of at most
                # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                # event loop for too long.
                if num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:
                    slices = (outputs.outputs, )
                else:
                    slices = np.array_split(
                        outputs.outputs,
                        cdiv(num_outputs, VLLM_V1_OUTPUT_PROC_CHUNK_SIZE))

                for i, outputs_slice in enumerate(slices):
                    # 2) Process EngineCoreOutputs.
                    processed_outputs = self.output_processor.process_outputs(
                        outputs_slice, outputs.timestamp, iteration_stats)
                    # NOTE: RequestOutputs are pushed to their queues.
                    assert not processed_outputs.request_outputs

                    # Allow other asyncio tasks to run between chunks
                    if i + 1 < len(slices):
                        await asyncio.sleep(0)

                    # 3) Abort any reqs that finished due to stop strings.
                    await self.engine_core.abort_requests_async(
                        processed_outputs.reqs_to_abort)

                # 4) Logging.
                # TODO(rob): make into a coroutine and launch it in
                # background thread once Prometheus overhead is non-trivial.
                self._record_stats(
                    engine_index=outputs.engine_index,
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                )

        except Exception as e:
            logger.exception("EngineCore output handler hit an error: %s", e)
            kill_process_tree(os.getpid())

    async def abort(self, request_id: str) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        request_ids = self.output_processor.abort_requests((request_id, ))
        await self.engine_core.abort_requests_async(request_ids)

        if self.log_requests:
            logger.info("Aborted request %s.", request_id)

    def _record_stats(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        engine_index: int = 0,
    ):
        if not self.log_stats:
            return

        assert scheduler_stats is not None
        for stat_logger in self.stat_loggers[engine_index]:
            stat_logger.record(scheduler_stats=scheduler_stats,
                               iteration_stats=iteration_stats)

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ):
        raise ValueError("Not Supported on V1 yet.")

    async def get_vllm_config(self) -> VllmConfig:
        return self.vllm_config

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def get_decoding_config(self):
        raise ValueError("Not Supported on V1 yet.")

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return self.processor.input_preprocessor

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return self.tokenizer.get_lora_tokenizer(lora_request)

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(
        self,
        scheduler_outputs=None,
        model_output=None,
    ) -> None:
        for loggers in self.stat_loggers:
            for stat_logger in loggers:
                stat_logger.log()

    async def check_health(self) -> None:
        logger.debug("Called check_health.")

    async def start_profile(self) -> None:
        await self.engine_core.profile_async(True)

    async def stop_profile(self) -> None:
        await self.engine_core.profile_async(False)

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        if device == Device.CPU:
            raise ValueError("Not supported on CPU.")
        await self.engine_core.reset_prefix_cache_async()

    async def sleep(self, level: int = 1) -> None:
        await self.engine_core.sleep_async(level)

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        await self.engine_core.wake_up_async(tags)

    async def is_sleeping(self) -> bool:
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return await self.engine_core.pin_lora_async(lora_id)

    @property
    def is_running(self) -> bool:
        return True

    @property
    def is_stopped(self) -> bool:
        return False

    @property
    def errored(self) -> bool:
        return False

    @property
    def dead_error(self) -> BaseException:
        return Exception()  # TODO: implement
