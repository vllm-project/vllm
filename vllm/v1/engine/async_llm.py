import asyncio
from typing import AsyncGenerator, Dict, List, Mapping, Optional, Type, Union

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
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_stream import AsyncStream
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.gpu_executor import GPUExecutor

logger = init_logger(__name__)


class AsyncLLM:

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

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            enable_lora=bool(vllm_config.lora_config))
        self.tokenizer.ping()

        # Map (request_id -> Stream)
        self.request_streams: Dict[str, AsyncStream] = {}

        # Processor (converts Inputs --> EngineCoreRequests)
        self.processor = Processor(vllm_config.model_config,
                                   vllm_config.lora_config, self.tokenizer,
                                   input_registry)

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput)
        self.detokenizer = Detokenizer(vllm_config.model_config.tokenizer)

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_client(
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

    @classmethod
    def _get_executor_cls(cls, engine_config: EngineConfig):
        return GPUExecutor

    def _add_request_to_streams(self, request_id: str) -> AsyncStream:
        if request_id in self.request_streams:
            raise ValueError(f"Request id {request_id} already running.")

        # TODO: handle abort.
        # IDEA(Nick): we could batch up aborts rather than sending
        # them individually, so that we send at most one batch of
        # aborts per step (added to any that we're doing due to
        # stop string matches for that step)
        def _abort():
            pass

        stream = AsyncStream(request_id, _abort)
        self.request_streams[request_id] = stream
        return stream

    def _send_to_streams(self, request_outputs: List[RequestOutput]):
        """Put the RequestOutputs into the corresponding AsyncStreams"""

        for request_output in request_outputs:
            request_id = request_output.request_id
            assert request_id in self.request_streams

            self.request_streams[request_id].put(request_output)

    async def abort_request(self, request_ids: List[str]) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        if request_ids:
            await self.engine_core.abort_requests_async(request_ids)
            self.detokenizer.abort_requests(request_ids)

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

        # 1) Make AsyncStream and add to self.request_streams.
        stream = self._add_request_to_streams(request_id)

        # 2) Convert input --> DetokenizerRequest / EngineCoreRequest.
        detokenizer_req, engine_core_req = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)

        # 3) Add the request to Detokenizer (this process).
        self.detokenizer.add_request(detokenizer_req)

        # 4) Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(engine_core_req)

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
        # TODO: shutdown remote worker execution loop (once TP enabled)

        logger.debug("Starting output handler busy loop in background loop.")

        try:
            while True:
                # Get EngineCoreOutput from the EngineCore.
                outputs = await self.engine_core.get_output_async()

                # Detokenize based on the output.
                request_outputs, reqs_to_abort = self.detokenizer.step(outputs)

                # Put the RequestOutputs into the per-request AsyncStream.
                # NOTE(rob): we could do the streaming in the detokenizer.
                self._send_to_streams(request_outputs)

                # Abort any requests that finished due to stop strings.
                await self.abort_request(reqs_to_abort)

        except BaseException as e:
            logger.error(e)
            raise e

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


# Retain V0 name for backwards compatibility.
AsyncLLMEngine = AsyncLLM
