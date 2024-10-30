import asyncio
from typing import AsyncGenerator, Dict, Mapping, Optional, Union

import zmq
import zmq.asyncio

from vllm.config import EngineConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.protocol import EngineClient
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import LLM_ENGINE_CORE_READY_STR
from vllm.v1.engine.async_stream import AsyncStream
from vllm.v1.engine.llm_engine import LLMEngine

logger = init_logger(__name__)

POLL_TIMEOUT_MS = 5000


class _AsyncLLMEngine(LLMEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, use_async_sockets=True)

    async def wait_for_startup(self):
        """Poll the ready socket until the LLMEngineCore is ready."""
        try:
            ready_socket = self.ctx.socket(zmq.constants.PULL)
            ready_socket.connect(self.readiness_ipc_path)
            while await ready_socket.poll(timeout=5000) == 0:
                logger.debug("Waiting for LLMEngineCore to startup.")

                if not self.engine_core.is_alive():
                    raise RuntimeError(
                        "LLMEngineCore process failed to start.")

            message = await ready_socket.recv_string()
            assert message == LLM_ENGINE_CORE_READY_STR

        finally:
            ready_socket.close(linger=0)

    async def add_request_async(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncStream:

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

        # 3) Add the request to EngineCore (separate process).
        await self.to_core.send_multipart(
            (self.encoder.encode(engine_core_request), ),
            copy=False,
            flags=zmq.NOBLOCK)

        return stream        


class AsyncLLMEngine(EngineClient):

    def __init__(self,
                 *args,
                 log_requests: bool = True,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:

        self.log_requests = log_requests
        self.engine = _AsyncLLMEngine(*args, **kwargs)

        assert start_engine_loop
        self.output_handler = asyncio.create_task(self.run_output_handler())

        # TODO: add background loop shielding
        # TODO: add AsyncEngineDeadError

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

        executor_class = LLMEngine._get_executor_cls(engine_config)

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
        """Wait until the _AsyncLLMEngine is ready"""

        await self.engine.wait_for_startup()

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

        if self.engine.detokenizer.is_request_active(request_id):
            raise KeyError(f"Request {request_id} already exists.")

        stream = await self.engine.add_request_async(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)

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
            while await self.engine.from_core.poll(timeout=POLL_TIMEOUT_MS) == 0:
                logger.debug("Waiting for output from LLMCore.")

            frames = await self.engine.from_core.recv_multipart(copy=False)
            engine_core_outputs = self.engine.decoder.decode(frames[0].buffer).outputs

            # Make RequestOutputs and push to the per-client output queues
            # NOTE: we could simplify the Detokenizer code by returning the full
            # List[RequestOutput] rather than pushing to the Queue at the expense
            # of doing another loop through List[RequestOutput] here.
            self.engine.detokenizer.step_streaming(engine_core_outputs)

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
