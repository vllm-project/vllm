# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from typing import AsyncGenerator, List, Mapping, Optional, Type, Union

import numpy as np

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.envs import VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
from vllm.inputs import INPUT_REGISTRY, InputRegistry, PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import cdiv, kill_process_tree
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import (ParallelSamplingOutputProcessor,
                                              ParentRequestState)
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
        executor_class: Type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        input_registry: InputRegistry = INPUT_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
    ) -> None:

        assert start_engine_loop

        self.model_config = vllm_config.model_config
        self.enable_prefix_caching = (
            vllm_config.cache_config.enable_prefix_caching)

        self.log_requests = log_requests
        self.log_stats = log_stats
        self.stat_loggers: List[StatLoggerBase] = [
            LoggingStatLogger(),
            PrometheusStatLogger(vllm_config.model_config),
        ]

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)
        self.tokenizer.ping()

        # Processor (converts Inputs --> EngineCoreRequests).
        self.processor = Processor(
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            lora_config=vllm_config.lora_config,
            tokenizer=self.tokenizer,
            input_registry=input_registry,
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
        )

        self.output_handler: Optional[asyncio.Task] = None

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[VllmConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        if engine_config is None:
            vllm_config = engine_args.create_engine_config(usage_context)
        else:
            vllm_config = engine_config

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
    ) -> asyncio.Queue[RequestOutput]:
        """Add new request to the AsyncLLM."""

        # 1) Create a new output queue for the request.
        if self.output_processor.is_request_active(request_id):
            raise ValueError(f"Request id {request_id} already running.")
        queue: asyncio.Queue[RequestOutput] = asyncio.Queue()

        # 2) Convert Input --> Request.
        request = self.processor.process_inputs(request_id, prompt, params,
                                                arrival_time, lora_request,
                                                trace_headers,
                                                prompt_adapter_request,
                                                priority)

        # 3) Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, queue)

        # 4) Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request_id)

        return queue

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def _generate(
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
                out = q.get_nowait() if not q.empty() else await q.get()

                # Coalesce any additional queued outputs
                while not q.empty():
                    next_out = q.get_nowait()
                    if sampling_params.output_kind == RequestOutputKind.DELTA:
                        out.add(next_out)
                    else:
                        out = next_out

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

    async def _parallel_sampling_task(
        self,
        gen: AsyncGenerator[RequestOutput, None],
        output_processor: ParallelSamplingOutputProcessor,
        index: int,
    ) -> AsyncGenerator[RequestOutput, None]:
        async for out in gen:
            if req_out := output_processor.process_output(out, index):
                yield req_out

    async def _generate_parallel_sampling(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        parent_state = ParentRequestState(request_id, sampling_params)
        output_processor = ParallelSamplingOutputProcessor(parent_state)
        n = parent_state.n

        # Adapted from sglang:
        # https://github.com/sgl-project/sglang/blob/
        # 4fe92bfca5517f3cf5ca967fc5fcfdb7cf335f30/
        # python/sglang/srt/managers/
        # tokenizer_manager.py#L456-L532

        if self.enable_prefix_caching:
            # If engine uses APC, generate a “warmup request” with
            # max_tokens=1 which populates the APC
            w_sampling_params = parent_state.get_child_sampling_params({
                "max_tokens":
                1,
                "n":
                1,
                "output_kind":
                RequestOutputKind.FINAL_ONLY
            })
            async for _ in self._generate(
                    prompt,
                    w_sampling_params,
                    parent_state.get_warmup_request_id(),
                    lora_request,
                    trace_headers,
                    prompt_adapter_request,
                    priority,
            ):
                # Exhaust the generator
                pass

        # Aggregate generators for n child requests
        gens = []
        active = {}
        seed = sampling_params.seed
        for idx in range(n):
            c_sampling_params = parent_state.get_child_sampling_params({
                "n":
                1,
                "seed":
                seed
            })
            if seed is not None:
                seed += 1
            child_gen = self._generate(
                prompt,
                c_sampling_params,
                parent_state.get_child_request_id(idx),
                lora_request,
                trace_headers,
                prompt_adapter_request,
                priority,
            )
            gen = self._parallel_sampling_task(child_gen, output_processor,
                                               idx)
            gens.append(gen)
            active[asyncio.create_task(gen.__anext__())] = idx

        try:
            while active:
                done, _ = await asyncio.wait(
                    active.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    idx = active.pop(task)
                    try:
                        result = task.result()
                        yield result
                        # Schedule the next result
                        active[asyncio.create_task(
                            gens[idx].__anext__())] = idx
                    except StopAsyncIteration:
                        continue
        finally:
            for task in active:
                task.cancel()

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
        n = sampling_params.n
        if n is None or sampling_params.n == 1:
            async for out in self._generate(prompt, sampling_params,
                                            request_id, lora_request,
                                            trace_headers,
                                            prompt_adapter_request, priority):
                yield out
        else:
            async for out in self._generate_parallel_sampling(
                    prompt, sampling_params, request_id, lora_request,
                    trace_headers, prompt_adapter_request, priority):
                yield out

    async def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        try:
            while True:
                # 1) Pull EngineCoreOutputs from the EngineCore.
                outputs = await self.engine_core.get_output_async()

                # Split outputs into chunks of at most
                # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                # event loop for too long.
                num_outputs = len(outputs.outputs)
                if num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:
                    slices = (outputs.outputs, )
                else:
                    slices = np.array_split(
                        outputs.outputs,
                        cdiv(num_outputs, VLLM_V1_OUTPUT_PROC_CHUNK_SIZE))

                iteration_stats = None
                for i, outputs_slice in enumerate(slices):
                    # 2) Process EngineCoreOutputs.
                    processed_outputs = self.output_processor.process_outputs(
                        outputs_slice, iteration_stats)
                    # NOTE: RequestOutputs are pushed to their queues.
                    assert not processed_outputs.request_outputs
                    iteration_stats = processed_outputs.iteration_stats

                    # Allow other asyncio tasks to run between chunks
                    if i + 1 < len(slices):
                        await asyncio.sleep(0)

                    # 3) Abort any reqs that finished due to stop strings.
                    await self.engine_core.abort_requests_async(
                        processed_outputs.reqs_to_abort)

                # 4) Logging.
                # TODO(rob): make into a coroutine and launch it in
                # background thread once Prometheus overhead is non-trivial.
                assert iteration_stats is not None
                self._log_stats(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                )

        except Exception as e:
            logger.exception("EngineCore output handler hit an error: %s", e)
            kill_process_tree(os.getpid())

    async def abort(self, request_id: str) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        request_ids = [request_id]
        await self.engine_core.abort_requests_async(request_ids)
        self.output_processor.abort_requests(request_ids)

        if self.log_requests:
            logger.info("Aborted request %s.", request_id)

    def _log_stats(
        self,
        scheduler_stats: SchedulerStats,
        iteration_stats: IterationStats,
    ):
        if not self.log_stats:
            return

        for logger in self.stat_loggers:
            logger.log(scheduler_stats=scheduler_stats,
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
        logger.debug("Called do_log_stats.")

    async def check_health(self) -> None:
        logger.debug("Called check_health.")

    async def start_profile(self) -> None:
        await self.engine_core.profile_async(True)

    async def stop_profile(self) -> None:
        await self.engine_core.profile_async(False)

    async def reset_prefix_cache(self) -> None:
        await self.engine_core.reset_prefix_cache_async()

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

    async def add_lora(self, lora_request: LoRARequest) -> None:
        """Load a new LoRA adapter into the engine for future requests."""
        raise NotImplementedError("LoRA not yet supported in V1")
