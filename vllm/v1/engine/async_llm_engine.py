import asyncio
import zmq
import zmq.asyncio
from typing import (AsyncGenerator, Dict, Optional, Mapping, Union)

from vllm.engine.protocol import EngineClient
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.async_stream import AsyncStream

logger = init_logger(__name__)

POLL_TIMEOUT_MS = 5000

class _AsyncLLMEngine(LLMEngine):
    def __init__(*args, **kwargs):
        super().__init__(*args, 
                         **kwargs,
                         use_async_sockets=True)

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
        stream = AsyncStream(request_id, _abort),

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

    async def step_async(self) -> None:

        while await self.from_core.poll(timeout=POLL_TIMEOUT_MS) == 0:
            logger.debug("Waiting for output from LLMCore.")
        frames = await self.from_core.recv_multipart(copy=False)
        engine_core_outputs = self.decoder.decode(frames[0].buffer).outputs

        # Make RequestOutputs and push to the per-client output queues
        # NOTE: we could simplify the Detokenizer code by returning the full
        # List[RequestOutput] rather than pushing to the Queue at the expense
        # of doing another loop through List[RequestOutput] here.        
        self.detokenizer.step_async(engine_core_outputs)

class AsyncLLMEngine(EngineClient):

    def __init__(
        self,
        *args,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        **kwargs
    ) -> None:
        self.log_requests = log_requests
        self.engine = _AsyncLLMEngine(*args, **kwargs)
        self.output_queues: Dict[str, asyncio.Queue] = {}
    
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

        return stream

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
            yield LLMEngine.validate_output(output, RequestOutput)

    async def run_engine_loop(self):
        # TODO: add weakref stuff in current AsyncLLMEngine
        # TODO: shutdown remote worker execution loop.
        # TODO: add PP

        while True:
            await self.engine.step_async()
