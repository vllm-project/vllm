import asyncio
import multiprocessing
from typing import AsyncGenerator, Dict, Mapping, Optional, Type, Union

import msgspec
import zmq
import zmq.asyncio

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
from vllm.utils import get_open_zmq_ipc_path
from vllm.v1.engine import LLM_ENGINE_CORE_READY_STR, EngineCoreOutputs
from vllm.v1.engine.async_stream import AsyncStream
from vllm.v1.engine.llm_engine_core import LLMEngineCore
from vllm.v1.engine.protocol import LLMEngineProtocol
from vllm.v1.executor.gpu_executor import GPUExecutor

logger = init_logger(__name__)

POLL_TIMEOUT_MS = 5000


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

        if usage_context == UsageContext.LLM_CLASS:
            scheduler_config.max_num_seqs = 1024
            scheduler_config.max_num_batched_tokens = 8192
        elif usage_context == UsageContext.OPENAI_API_SERVER:
            scheduler_config.max_num_seqs = 1024
            scheduler_config.max_num_batched_tokens = 2048

        super().__init__(
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
            executor_class,
            log_stats,
            usage_context,
            stat_loggers,
            input_registry,
            use_cached_outputs,
        )
        self.detokenizer.stream_mode = True
        self.log_requests = log_requests

        # IPC Setup
        self.ctx = zmq.asyncio.Context()  # type: ignore[attr-defined]
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(EngineCoreOutputs)

        # Path for IPC.
        self.ready_path = get_open_zmq_ipc_path()

        # Get output (EngineCoreOutput) from LLMEngineCore.
        output_path = get_open_zmq_ipc_path()
        self.output_socket = self.ctx.socket(zmq.constants.PULL)
        self.output_socket.connect(output_path)

        # Send input (EngineCoreRequest) to LLMEngineCore.
        input_path = get_open_zmq_ipc_path()
        self.input_socket = self.ctx.socket(zmq.constants.PUSH)
        self.input_socket.bind(input_path)

        # The current process might have CUDA context,
        # so we need to spawn a new process
        context = multiprocessing.get_context("spawn")

        # Run LLMEngineCore busy loop in background process.
        self.engine_core_process = context.Process(target=self.run_engine_core,
                                                   args=(
                                                       executor_class,
                                                       model_config,
                                                       cache_config,
                                                       parallel_config,
                                                       scheduler_config,
                                                       device_config,
                                                       load_config,
                                                       lora_config,
                                                       speculative_config,
                                                       observability_config,
                                                       prompt_adapter_config,
                                                   ),
                                                   kwargs={
                                                       "async_mode":
                                                       True,
                                                       "input_path":
                                                       input_path,
                                                       "output_path":
                                                       output_path,
                                                       "ready_path":
                                                       self.ready_path,
                                                   })

        # TODO: add background loop shielding
        # TODO: add AsyncEngineDeadError
        self.output_handler = asyncio.create_task(self.run_output_handler())

    def __del__(self):
        # Hack.
        self.engine_core_process.kill()

    def run_engine_core(self, *args, **kwargs):
        """Launch EngineCore busy loop in background process."""

        logger.debug("Initializing LLMEngineCore in background process.")
        engine_core = LLMEngineCore(*args, **kwargs)

        logger.debug("Starting LLMEngineCore busy loop in bavkgroudn process.")
        engine_core.run_busy_loop()

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

        # Create the async LLM engine.
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

    async def wait_for_startup(self):
        """Poll the ready socket until the LLMEngineCore is ready."""

        try:
            ready_socket = self.ctx.socket(zmq.constants.PULL)
            ready_socket.connect(self.ready_path)
            while await ready_socket.poll(timeout=5000) == 0:
                logger.debug("Waiting for LLMEngineCore to startup.")

                if not self.engine_core_process.is_alive():
                    raise RuntimeError(
                        "LLMEngineCore process failed to start.")

            message = await ready_socket.recv_string()
            assert message == LLM_ENGINE_CORE_READY_STR

        finally:
            ready_socket.close(linger=0)

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

        # 1) Process raw inputs into the request.
        detokenizer_request, engine_core_request = self._process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)

        # 2) Add the request to Detokenizer (this process).
        self.detokenizer.add_request(detokenizer_request, stream)

        # 3) Add the EngineCoreRequest to EngineCore (separate process).
        await self.input_socket.send_multipart(
            (self.encoder.encode(engine_core_request), ),
            copy=False,
            flags=zmq.NOBLOCK)

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
        # TODO: shutdown remote worker execution loop.

        while True:
            while await self.output_socket.poll(timeout=POLL_TIMEOUT_MS) == 0:
                logger.debug("Waiting for output from LLMCore.")

            frames = await self.output_socket.recv_multipart(copy=False)
            engine_core_outputs = self.decoder.decode(frames[0].buffer).outputs

            # Make RequestOutputs and push to the per-client output queues
            # NOTE: we could simplify the Detokenizer code by returning the full
            # List[RequestOutput] rather than pushing to the Queue at the
            # expense of doing another loop through List[RequestOutput] here.
            self.detokenizer.step_streaming(engine_core_outputs)

    async def abort(self):
        pass

    async def check_health(self):
        pass

    async def dead_error(self):
        pass

    async def do_log_stats(self):
        pass

    async def encode(self):
        pass

    async def errored(self):
        pass

    async def get_decoding_config(self):
        pass

    async def get_model_config(self):
        pass

    async def get_tokenizer(self):
        pass

    async def is_running(self):
        pass

    async def is_stopped(self):
        pass

    async def is_tracing_enabled(self):
        pass

    async def start_profile(self):
        pass

    async def stop_profile(self):
        pass
