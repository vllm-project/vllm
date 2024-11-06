import asyncio
from functools import partial
from typing import (AsyncGenerator, Dict, List, Mapping, Optional, Type,
                    Union)

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.protocol import EngineClient
from vllm.inputs import INPUT_REGISTRY, InputRegistry, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_stream import AsyncStream
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.gpu_executor import GPUExecutor

logger = init_logger(__name__)


class AsyncLLM(EngineClient):

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[GPUExecutor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
    ) -> None:
        assert start_engine_loop

        self.log_requests = log_requests
        self.log_stats = log_stats
        self.stat_loggers = stat_loggers
        self.model_config = vllm_config.model_config

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            enable_lora=bool(vllm_config.lora_config))
        self.tokenizer.ping()

        # Request streams (map of request_id -> AsyncStream).
        self.request_streams: Dict[str, AsyncStream] = {}

        # Processor (converts Inputs --> EngineCoreRequests).
        self.processor = Processor(vllm_config.model_config,
                                   vllm_config.lora_config, self.tokenizer,
                                   input_registry)

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput).
        self.detokenizer = Detokenizer(vllm_config.model_config.tokenizer)

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            usage_context=usage_context,
            multiprocess_mode=True,
            asyncio_mode=True,
        )

        self.is_output_handler_running = False

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[VllmConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "AsyncLLMEngine":
        """Creates an AsyncLLMEngine from the EngineArgs."""

        # Create the engine configs.
        if engine_config is None:
            vllm_config = engine_args.create_engine_config()
        else:
            vllm_config = engine_config

        executor_class = cls._get_executor_cls(vllm_config)

        # Create the AsyncLLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def shutdown(self):
        self.engine_core.shutdown()

        if hasattr(self, "output_handler"):
            self.output_handler.cancel()

    @classmethod
    def _get_executor_cls(cls, vllm_config: VllmConfig):
        return GPUExecutor

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
    ) -> AsyncGenerator[Union[RequestOutput, EmbeddingRequestOutput], None]:
        """Add request_id to the EngineCore and return a Generator."""

        if self.detokenizer.is_request_active(request_id):
            raise KeyError(f"Request {request_id} already exists.")

        # 1) Create a new request in the RequestTracker.
        stream = self._add_request_to_streams(request_id,
                                              verbose=self.log_requests)

        # 2) Convert input --> DetokenizerRequest / EngineCoreRequest.
        detokenizer_req, engine_core_req = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)

        # 3) Add the request to Detokenizer (this process).
        self.detokenizer.add_request(detokenizer_req)

        # 4) Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(engine_core_req)

        # 5) Return the generator.
        return stream.generator()


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

        # We start the output_handler on the first call to generate() so that
        # we can call __init__ before the event loop starts, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        if not self.is_output_handler_running:
            self.output_handler = asyncio.create_task(
                self._run_output_handler())
            self.is_output_handler_running = True

        async for output in await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
        ):
            yield output


    async def _abort_requests(
        self,
        request_ids: Union[str, List[str]],
        *,
        exception: Optional[Union[BaseException, Type[BaseException]]] = None,
        verbose: bool,
    ) -> None:
        """
        Abort requests. This function is called in two places:
            * In output_handler_loop, if stop string is detected
            * In iterate_with_cancellation (inside AsyncStream) when
                a request disconnects. This function is as a callback.
        """

        # Convert to a list if we got a single request_id str.
        if isinstance(request_ids, str):
            request_ids = [request_ids]

        # Abort in EngineCore and Detokenizer.
        await self.engine_core.abort_requests_async(request_ids)
        self.detokenizer.abort_requests(request_ids)

        # Remove from the request streams.
        for request_id in request_ids:
            stream = self.request_streams.pop(request_id, None)
            if stream is not None:
                stream.finish(exception=exception)

            if verbose:
                logger.info("Aborted request %s.", request_id)

    def _add_request_to_streams(
        self,
        request_id: str,
        verbose: bool = False,
    ) -> AsyncStream:
        """Add a request to the request streams."""

        if request_id in self.request_streams:
            raise ValueError(f"Request id {request_id} already running.")

        abort_callback = partial(self._abort_requests, verbose=verbose)
        stream = AsyncStream(request_id, abort_callback)
        self.request_streams[request_id] = stream

        if verbose:
            logger.info("Added request %s.", request_id)

        return stream

    def _process_request_outputs(self, request_outputs: List[RequestOutput]):
        """Put the outputs in streams and remove from tracker if finished."""

        for request_output in request_outputs:
            request_id = request_output.request_id
            assert request_id in self.request_streams

            # Each request in the API server pulls from these streams.
            self.request_streams[request_id].put(request_output)

            # If finished, remove from the tracker.
            if request_output.finished:
                self.request_streams[request_id].finish()
                self.request_streams.pop(request_id)

    async def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        try:
            while True:
                # 1) Pull EngineCoreOutput from the EngineCore.
                outputs = await self.engine_core.get_output_async()

                # 2) Detokenize based on the output.
                request_outputs, reqs_to_abort = self.detokenizer.step(outputs)

                # 3) Put the RequestOutputs into the per-request AsyncStreams.
                self._process_request_outputs(request_outputs)

                # Abort any requests that finished due to stop strings.
                await self._abort_requests(reqs_to_abort, verbose=False)

        except BaseException as e:
            logger.error(e)
            raise e

    # TODO: can we eliminate these?

    async def abort(self, request_id: str) -> None:
        # Note: Who Calls this? I dont think this is actually used.
        raise ValueError("Not Supported on V1 yet.")

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

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        assert lora_request is None
        return self.detokenizer.tokenizer

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(
        self,
        scheduler_outputs=None,
        model_output=None,
    ) -> None:
        logger.debug("Called do_log_stats.")

    async def check_health(self) -> None:
        logger.debug("Called do_log_stats.")

    async def start_profile(self) -> None:
        raise ValueError("Not supported on V1 yet.")

    async def stop_profile(self) -> None:
        raise ValueError("Not supported on V1 yet.")

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
        return Exception


# Retain V0 name for backwards compatibility.
AsyncLLMEngine = AsyncLLM
