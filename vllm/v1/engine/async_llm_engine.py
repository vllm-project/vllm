import asyncio
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
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.llm_engine_core import LLMEngineCoreProcess
from vllm.v1.engine.processor import Processor
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
        assert start_engine_loop

        if usage_context == UsageContext.LLM_CLASS:
            scheduler_config.max_num_seqs = 1024
            scheduler_config.max_num_batched_tokens = 8192
        elif usage_context == UsageContext.OPENAI_API_SERVER:
            scheduler_config.max_num_seqs = 1024
            scheduler_config.max_num_batched_tokens = 2048

        self.log_requests = log_requests

        # Processor (convert Inputs --> EngineCoreRequests)
        self.processor = Processor(model_config, parallel_config,
                                   scheduler_config, lora_config,
                                   input_registry)

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput)
        self.detokenizer = Detokenizer(model_config.tokenizer,
                                       stream_mode=True)

        # IPC Setup
        self.ctx = zmq.asyncio.Context()  # type: ignore[attr-defined]
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(EngineCoreOutputs)

        # Path for IPC.
        ready_path = get_open_zmq_ipc_path()
        output_path = get_open_zmq_ipc_path()
        input_path = get_open_zmq_ipc_path()

        # Get output (EngineCoreOutput) from LLMEngineCore.
        self.output_socket = self.ctx.socket(zmq.constants.PULL)
        self.output_socket.connect(output_path)

        # Send input (EngineCoreRequest) to LLMEngineCore.
        self.input_socket = self.ctx.socket(zmq.constants.PUSH)
        self.input_socket.bind(input_path)

        self.engine_core = LLMEngineCoreProcess.from_config(
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
            input_path=input_path,
            output_path=output_path,
            ready_path=ready_path,
        )
        self.engine_core.start()
        self.wait_for_engine_core(ready_path)

        # TODO: add background loop shielding
        # TODO: add AsyncEngineDeadError
        self.output_handler = asyncio.create_task(self.run_output_handler())

    def __del__(self):
        # Hack.
        self.engine_core.kill()

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

    def wait_for_engine_core(self, ready_path: str):
        """Wait until the LLMEngineCore is ready."""

        try:
            # Non-asyncio context so this can run in __init__
            sync_ctx = zmq.Context()  # type: ignore[attr-defined]
            socket = sync_ctx.socket(zmq.constants.PULL)
            socket.connect(ready_path)

            # Poll ready socket socket until
            while socket.poll(timeout=POLL_TIMEOUT_MS) == 0:
                logger.debug("Waiting for LLMEngineCore to startup.")

                if not self.engine_core.is_alive():
                    raise RuntimeError(
                        "LLMEngineCore process failed to start.")

            message = socket.recv_string()
            assert message == LLM_ENGINE_CORE_READY_STR

        except BaseException as e:
            logger.exception(e)
            raise e

        finally:
            sync_ctx.destroy(linger=0)

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
        detokenizer_req, engine_core_req = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)

        # 2) Add the request to Detokenizer (this process).
        self.detokenizer.add_request(detokenizer_req, stream)

        # 3) Add the EngineCoreRequest to EngineCore (separate process).
        await self.input_socket.send_multipart(
            (self.encoder.encode(engine_core_req), ),
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
        # TODO: shutdown remote worker execution loop

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
