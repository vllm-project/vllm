import asyncio
from typing import (Any, AsyncGenerator, Callable, Dict, Mapping, Optional,
                    Type, Union)

from vllm.config import EngineConfig, ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.inputs import INPUT_REGISTRY, InputRegistry, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.gpu_executor import GPUExecutor

logger = init_logger(__name__)


class AsyncLLMEngine:

    def __init__(
        self,
        vllm_config: EngineConfig,
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
        self.errored = False

        # Processor (converts Inputs --> EngineCoreRequests)
        self.processor = Processor(vllm_config.model_config,
                                   vllm_config.lora_config, self.tokenizer,
                                   input_registry)

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput)
        self.detokenizer = Detokenizer(vllm_config.model_config.tokenizer,
                                       stream_mode=True)

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient(
            vllm_config=vllm_config,
            executor_class=executor_class,
            usage_context=usage_context,
            multiprocess_mode=True,
            asyncio_mode=True,
        )

        # TODO: add background loop shielding
        # TODO: add AsyncEngineDeadError

        self.is_output_handler_running = False

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[EngineConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "AsyncLLMEngine":
        """Creates an AsyncLLMEngine from the EngineArgs."""

        # Create the engine configs.
        if engine_config is None:
            engine_config = engine_args.create_engine_config()

        executor_class = cls._get_executor_cls(engine_config)

        # Create the AsyncLLMEngine.
        return cls(
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    @classmethod
    def _get_executor_cls(cls, engine_config: EngineConfig):
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

        if self.detokenizer.is_request_active(request_id):
            raise KeyError(f"Request {request_id} already exists.")

        # TODO: handle abort.
        # IDEA(Nick): we could batch up aborts rather than sending
        # them individually, so that we send at most one batch of
        # aborts per step (added to any that we're doing due to
        # stop string matches for that step)
        def _abort():
            pass

        # AsyncStream generator
        stream = AsyncStream(request_id, _abort)

        # 1) Convert input --> DetokenizerRequest / EngineCoreRequest.
        detokenizer_req, engine_core_req = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)

        # 2) Add the request to Detokenizer (this process).
        self.detokenizer.add_request(detokenizer_req, stream)

        # 3) Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(engine_core_req)

        logger.debug("Added request %s.", request_id)

        return stream.generator()

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

    async def _run_output_handler(self):
        # TODO: add weakref from current AsyncLLMEngine
        # TODO: shutdown remote worker execution loop

        logger.debug("Starting output handler busy loop in background loop.")

        try:
            while True:
                outputs = await self.engine_core.get_output_async()

                # Make RequestOutputs and push to the per-client output queues
                # NOTE: we could simplify the Detokenizer code by returning full
                # List[RequestOutput] rather than pushing to the Queue at the
                # expense of doing another loop through List[RequestOutput].
                _to_abort = self.detokenizer.step_streaming(outputs)

                # TODO: send aborts (in one message)
        except BaseException as e:
            logger.error(e)

    # TODO: can we eliminate these (used by OpenAI server)

    async def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        assert lora_request is None
        return self.detokenizer.tokenizer

    async def is_tracing_enabled(self) -> bool:
        return False


class AsyncStream:
    """A stream of RequestOutputs or EmbeddingRequestOutputs for a request
    that can be iterated over asynchronously via an async generator."""

    STOP_ITERATION = Exception()  # Sentinel

    def __init__(self, request_id: str, cancel: Callable[[str], None]) -> None:
        self.request_id = request_id
        self._cancel = cancel
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, EmbeddingRequestOutput,
                              Exception]) -> None:
        if not self._finished:
            self._queue.put_nowait(item)

    def finish(
        self,
        exception: Optional[Union[BaseException, Type[BaseException]]] = None,
    ) -> None:
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(exception if self._is_raisable(exception)
                                   else AsyncStream.STOP_ITERATION)

    @property
    def finished(self) -> bool:
        return self._finished

    async def generator(
        self
    ) -> AsyncGenerator[Union[RequestOutput, EmbeddingRequestOutput], None]:
        try:
            while True:
                result = await self._queue.get()
                if self._is_raisable(result):
                    if result == AsyncStream.STOP_ITERATION:
                        return
                    raise result
                yield result
        except GeneratorExit:
            self._cancel(self.request_id)
            raise asyncio.CancelledError from None

    @staticmethod
    def _is_raisable(value: Any):
        return isinstance(value, BaseException) or \
                (isinstance(value, type) and \
                 issubclass(value, BaseException))
