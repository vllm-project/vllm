import asyncio
from typing import AsyncGenerator, Dict, Mapping, Optional, Type, Union

from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig,
                         EngineConfig, LoadConfig, LoRAConfig, ModelConfig,
                         ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.inputs import INPUT_REGISTRY, InputRegistry, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_stream import AsyncStream
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.core import EngineCoreClient
from vllm.v1.engine.processor import Processor
from vllm.v1.engine.protocol import LLMEngineProtocol
from vllm.v1.executor.gpu_executor import GPUExecutor

logger = init_logger(__name__)


class AsyncLLMEngine(LLMEngineProtocol):

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        speculative_config: Optional[SpeculativeConfig],
        decoding_config: Optional[DecodingConfig],
        observability_config: Optional[ObservabilityConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
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

        # Processor (converts Inputs --> EngineCoreRequests)
        self.processor = Processor(model_config, parallel_config,
                                   scheduler_config, lora_config,
                                   input_registry)

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput)
        self.detokenizer = Detokenizer(model_config.tokenizer,
                                       stream_mode=True)

        # EngineCore (starts the engine in background process).
        self.engine_core_client = EngineCoreClient(
            executor_class,
            model_config,
            cache_config,
            parallel_config,
            scheduler_config,
            device_config,
            load_config,
            lora_config,
            speculative_config,
            decoding_config,
            observability_config,
            prompt_adapter_config,
            usage_context,
        )

        # TODO: add background loop shielding
        # TODO: add AsyncEngineDeadError
        self.output_handler = asyncio.create_task(self.run_output_handler())

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
        engine = cls(
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )
        return engine

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
        await self.engine_core_client.add_request_async(engine_core_req)

        logger.debug("Added request %s.", request_id)

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

    async def run_output_handler(self):
        # TODO: add weakref from current AsyncLLMEngine
        # TODO: shutdown remote worker execution loop

        logger.debug("Starting output handler busy loop in background loop.")

        try:
            while True:
                logger.debug("About call get_output_async")
                outputs = await self.engine_core_client.get_output_async()
                logger.debug("Got EngineRequestOutput from EngineCore")

                # Make RequestOutputs and push to the per-client output queues
                # NOTE: we could simplify the Detokenizer code by returning full
                # List[RequestOutput] rather than pushing to the Queue at the
                # expense of doing another loop through List[RequestOutput].
                self.detokenizer.step_streaming(outputs)
                logger.debug("Stepped the Detokenizer.")
        except BaseException as e:
            logger.error(e)
