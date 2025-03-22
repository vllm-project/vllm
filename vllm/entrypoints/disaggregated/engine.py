# SPDX-License-Identifier: Apache-2.0

import asyncio
import msgspec
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Dict, Mapping, Optional

import uvicorn
import uvloop
import zmq
import zmq.asyncio

from vllm import SamplingParams
from vllm.config import DecodingConfig, ModelConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.inputs.data import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import Device, FlexibleArgumentParser, make_zmq_socket

logger = init_logger(__name__)

DEFAULT_MAX_TOKENS = 32000

# NOTE FOR DEVELOPERS:
# DO NOT USE PICKLE FOR THESE CLASSES. IN A MULTI NODE
# SETUP WE WILL USE TCP. WE CANNOT USE PICKLE OTHERWISE
# WE RISK REMOTE CODE EXECUTION FROM UNSTRUSTED USERS.

class PDRequest(msgspec.Struct,
              array_like=True,  # type: ignore[call-arg]
              omit_defaults=True,  # type: ignore[call-arg]
              gc=False):  # type: ignore[call-arg]
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    # TODO: support multimodal inputs.

class PDResponse(msgspec.Struct,
              array_like=True,  # type: ignore[call-arg]
              omit_defaults=True,  # type: ignore[call-arg]
              gc=False):  # type: ignore[call-arg]
    request_id: str
    success: bool
    delta_text: Optional[str] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    logprobs = None # TODO

@asynccontextmanager
def build_pd_engine_client(prefill_addr: str, decode_addr: str,
                            connector_addr: str):
    engine = PDEngine(prefill_addr, decode_addr, connector_addr)
    yield engine
    engine.shutdown()

class PDEngine:
    """
    PDEngine:
        Equiavlent of AsyncLLM for P/D. Assumes there is
        a Prefill and Decode service already running.

        * TODO: actually handle errors and failure.
        * TODO: support more than just text input.
        * TODO: move under vllm/v1/engine one past prototype.
    """

    def __init__(self, prefill_addr: str, decode_addr: str, connector_addr: str):
        # Request queues.
        self.queues: Dict[str, asyncio.Queue] = {}

        # Serialization encoder.
        self.encoder = msgspec.msgpack.Encoder()

        # ZMQ communication..
        self.ctx = zmq.asyncio.Context()
        self.to_decode = make_zmq_socket(
            self.ctx, f"{decode_addr}", zmq.constants.PUSH)
        self.to_prefill = make_zmq_socket(
            self.ctx, f"{prefill_addr}", zmq.constants.PUSH)
        self.connector_addr = connector_addr
        
        # Background loops (started on first generate()).
        self.output_handler: Optional[asyncio.Task] = None
        self.log_running: Optional[asyncio.Task] = None

    def shutdown(self):
        if (ctx := self.ctx) is not None:
            ctx.destroy(linger=0)
        if (task := self.log_running) is not None:
            task.cancel()
        if (task := self.output_handler) is not None:
            task.cancel()

    async def _run_log_running(self):
        logger.info("Running requests: %d", len(self.queues))
        await asyncio.sleep(10.)

    async def _run_output_handler(self, socket: zmq.asyncio.Socket):
        """
        Pull responses from Decode + Prefill engines and
        distribute back to the generate() tasks.
        """
        decoder = msgspec.msgpack.Decoder(PDResponse)
        
        socket: Optional[zmq.asyncio.Socket] = None
        try:
            socket = make_zmq_socket(
                self.ctx, self.connector_addr, zmq.constants.PULL)

            while True:
                reponse_bytes = await socket.recv().buffer
                response = decoder.decode(reponse_bytes)
                self.queues[response.request_id].put_nowait(response)
        except:
            # TODO: actually handle failure and shutdown.
            raise 
        finally:
            if socket is not None:
                socket.close(linger=0)
    
    async def _prefill(self,
                       request: PDRequest,
                       q: asyncio.Queue[PDResponse]) -> PDResponse:
        # Send request to the prefill instance.
        req_bytes = self.encoder(request)
        await self.to_prefill.send(req_bytes, copy=False)

        # Wait for the prefill to be done.
        response = await q.get()
        assert response.request_id == request.request_id
        if not response.success:
            # TODO: actual error handling and shutdown.
            raise Exception("Failed Prefill Request.")

        return response
    
    async def _decode(self,
                      request: PDRequest,
                      q: asyncio.Queue[PDResponse]) -> AsyncGenerator[PDResponse]:
        # Send request to the decode instance.
        req_bytes = self.encoder(request)
        await self.to_decode.send(req_bytes, copy=False)

        # Iterate response queue and yield each response to caller..
        finished = False
        while not finished:
            response = await q.get()
            if not response.success:
                # TODO: actual error handling and shutdown.
                raise Exception("Failed Decode Request.")
            finished = response.finish_reason is not None
            yield response

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PDResponse]:
        # Start loops on first request.
        if self.output_handler is None:
            self.output_handler = asyncio.create_task(self._run_output_handler())
            self.log_running = asyncio.create_task(self._run_log_running())

        # TODO: expand to suppo
        if not isinstance(prompt, str):
            raise ValueError("We currently only support text inputs!")
        if request_id in self.queues:
            raise ValueError(f"Found duplicate request_id: {request_id}!")
        
        # Queue to gather output from output_handler.
        q: asyncio.Queue[PDResponse] = asyncio.Queue()
        self.queues[request_id] = q

        # (1) Perform the prefill (max_tokens=1).
        original_max_tokens = sampling_params.max_tokens
        request = PDRequest(request_id, prompt, sampling_params)
        request.sampling_params.max_tokens = 1
        response = await self._prefill(request, q)
        yield response

        # (2) Perform the decodes (original tokens).
        request.sampling_params.max_tokens = original_max_tokens
        async for response in self._decode(request, q):
            yield response

    async def beam_search(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:
        raise NotImplementedError

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        raise NotImplementedError

    async def abort(self, request_id: str) -> None:
        raise NotImplementedError

    async def get_model_config(self) -> ModelConfig:
        raise NotImplementedError

    async def get_decoding_config(self) -> DecodingConfig:
        raise NotImplementedError

    async def get_input_preprocessor(self) -> InputPreprocessor:
        raise NotImplementedError

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        raise NotImplementedError

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[List[SamplerOutput]] = None,
    ) -> None:
        pass

    async def check_health(self) -> None:
        pass

    async def start_profile(self) -> None:
        raise NotImplementedError

    async def stop_profile(self) -> None:
        raise NotImplementedError

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        raise NotImplementedError

    async def sleep(self, level: int = 1) -> None:
        raise NotImplementedError

    async def wake_up(self) -> None:
        raise NotImplementedError

    async def is_sleeping(self) -> bool:
        False

    async def add_lora(self, lora_request: LoRARequest) -> None:
        raise NotImplementedError
    

async def run_disagg_connector(args, **uvicorn_kwargs):
    logger.info("vLLM Connector Start: %s %s", args, uvicorn_kwargs)

    # NOTE FOR DEVELOPERS: when we shift this to TCP, we must
    # ensure that the serialization is not pickle based to
    # avoid RCE issues from untrusted users!!!
    app.state.port = args.port
    app.state.connector_addr = f"ipc://{args.connector_addr}"
    app.state.decode_addr = f"ipc://{args.decode_addr}"
    app.state.prefill_addr = f"ipc://{args.prefill_addr}"

    # init uvicorn server
    config = uvicorn.Config(app, host=args.post, port=args.port)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--connector-addr",
                        type=str,
                        required=True,
                        help="The zmq ipc connector address")
    parser.add_argument("--prefill-addr",
                        type=str,
                        required=True,
                        help="The zmq ipc prefill address")
    parser.add_argument("--decode-addr",
                        type=str,
                        required=True,
                        help="The zmq ipc decode address")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
