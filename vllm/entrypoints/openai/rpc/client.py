import asyncio
from contextlib import contextmanager
from typing import Any, AsyncGenerator, Mapping, Optional
from uuid import uuid4

import cloudpickle
import zmq
import zmq.asyncio

from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.entrypoints.openai.rpc import (RPC_REQUEST_TYPE,
                                         VLLM_RPC_HEALTH_TIMEOUT_MS,
                                         VLLM_RPC_SERVER_START_TIMEOUT_MS,
                                         VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCGenerateRequest, RPCUtilityRequest)
from vllm.inputs import PromptInputs
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

logger = init_logger(__name__)


# Path used for inprocess proxy.
INPROC_PROXY_PATH = f"inproc://{uuid4()}"


class AsyncEngineRPCClient:
    """
    RPCClient that connects to the RPCServer wrapping AsyncLLMEngine.
    
    On startup, the RPCClient:
        - makes DEALER socket (to_rpc_server) that connects to the RPCServer 
            via ipc, which uses unix sockets under the hood
            (https://libzmq.readthedocs.io/en/zeromq4-1/zmq_ipc.html)
        - makes ROUTER socket (from_api_server) that binds to a random 
            inproc address, which uses memory under the hood
            (https://libzmq.readthedocs.io/en/zeromq3-x/zmq_inproc.html)
        - runs a proxy in a background asyncio task between 
            from_api_server (ROUTER, inproc) and to_rpc_server (DEALER ipc, )

    Each request handled by the asyncio api_server calls generate():
        - make a DEALER socket that connects to from_api_server via inproc
        - send a RCPGenerateRequest to the inproc socket
        - background proxy forwards the request from inproc -> ipc
        - RPCServer responds to the request one token at a time over ipc
        - background proxy forwards the response from ipc -> inproc

    The connection looks like this:
        DEALER <- inproc -> [ ROUTER | DEALER ] <- ipc -> ROUTER
    
    Message routing is performed via identities that are managed by the 
    ROUTER socket. ROUTER sockets track every connection it has and 
    tells the caller about these. The way it tells the caller is to stick 
    the connection identity in front of each message received. When we 
    send the message via a ROUTER, we first send an identity frame.
    See https://zguide.zeromq.org/docs/chapter3/#The-Extended-Reply-Envelope
    for more details on connection identities.

    This proxy design enables us to use a single unix socket, which 
    improves performance by avoiding syscalls (~5%) and avoids resource limits
    such as ulimit, which defaults to 1024 on ubuntu.

    See: https://zguide.zeromq.org/docs/chapter3/ for more details on the
    Request-Reply pattern of zeromq sockets.
    """

    def __init__(self, rpc_path: str):
        self.context = zmq.asyncio.Context()
        self.context.set(zmq.constants.MAX_SOCKETS,
                         self.context.get(zmq.constants.SOCKET_LIMIT))

        # IPC connection to RPC Server (uses unix sockets).
        self.to_rcp_server = self.context.socket(zmq.constants.DEALER)
        self.to_rcp_server.connect(rpc_path)

        # In process proxy to RPC Server (used memory-based messaging).
        self.from_api_server = self.context.socket(zmq.constants.ROUTER)
        self.from_api_server.bind(INPROC_PROXY_PATH)

        # Asyncio background task for the proxy.
        self.proxy_task = asyncio.create_task(
            self.run_proxy(self.from_api_server, self.to_rcp_server))

        # Maximum number of requests that can be active. This value is
        # used uvicorn to launch with --limit-concurrency to limit the
        # maximum number of requests being processed at a time.
        # Note: https://www.uvicorn.org/server-behavior/#resource-limits
        # Note: this value is typically 65536
        self.limit_concurrency = self.context.get(zmq.constants.SOCKET_LIMIT)

    @property
    def is_running(self) -> bool:
        return not self._errored

    @property
    def is_stopped(self) -> bool:
        return self._errored

    @property
    def errored(self) -> bool:
        return self._errored
    
    def close(self):
        """Destroy the ZeroMQ Context."""
        self.from_api_server.close()
        self.to_rcp_server.close()
        self.context.destroy()
    
    async def run_proxy(self, socket_from, socket_to):
        """Background task that runs a proxy"""
        poller = zmq.asyncio.Poller()
        poller.register(socket_from, zmq.constants.POLLIN)
        poller.register(socket_to, zmq.constants.POLLIN)
        while True:
            events = await poller.poll()
            events = dict(events)
            if socket_from in events:
                msg = await socket_from.recv_multipart()
                await socket_to.send_multipart(msg)
            if socket_to in events:
                msg = await socket_to.recv_multipart()
                await socket_from.send_multipart(msg)

    async def setup(self):
        """Setup the client before it starts sending server requests.
        
        This should be called immediately after __init__
        (it would be part of __init__ if not for async)
        """

        # Wait until server is ready.
        await self._wait_for_server_rpc()
        self._errored = False

        # Get the configs.
        self.model_config = await self._get_model_config_rpc()
        self.decoding_config = await self._get_decoding_config_rpc()
        self.tracing_flag = await self._is_tracing_enabled_rpc()

        # Create the tokenizer group.
        # TODO: refactor OAI server to avoid needing this info.
        self.tokenizer = init_tokenizer_from_configs(
            model_config=self.model_config,
            scheduler_config=(await self._get_scheduler_config_rpc()),
            parallel_config=(await self._get_parallel_config_rpc()),
            enable_lora=bool(await self._get_lora_config_rpc()),
        )

    @contextmanager
    def to_proxy_socket(self):
        # Connect to the proxy.
        # DEALER enables asynch communication for streaming.
        socket = self.context.socket(zmq.constants.DEALER)
        try:
            socket.connect(INPROC_PROXY_PATH)
            yield socket
        finally:
            # linger == 0 means discard unsent messages
            # when the socket is closed. This is necessary
            # because otherwise self.context.destroy() will
            # wait for 30 seconds until unsent messages are
            # received, which is impossible if the server
            # crashed. In the absence of a server crash we
            # always expect a response before closing the
            # socket anyway.
            # Reference: http://api.zeromq.org/4-2:zmq-setsockopt#toc24
            socket.close(linger=0)

    async def _send_get_data_rpc_request(self, request: RPCUtilityRequest,
                                         expected_type: Any,
                                         error_message: str) -> Any:
        """Send an RPC request that is expecting data back."""

        with self.to_proxy_socket() as socket:
            # Ping RPCServer with a request.
            await socket.send_multipart([cloudpickle.dumps(request)])

            # Await the data from the Server.
            response = cloudpickle.loads(await socket.recv())

        if not isinstance(response, expected_type):
            # LoRAConfig can be None.
            if expected_type == LoRAConfig and response is None:
                pass
            # Propogate Exception Engine.
            elif isinstance(response, Exception):
                logger.warning(error_message)
                raise response
            else:
                raise ValueError(error_message)

        return response

    async def _send_one_way_rpc_request(self,
                                        request: RPC_REQUEST_TYPE,
                                        error_message: str,
                                        timeout: Optional[int] = None):
        """Send one-way RPC request to trigger an action."""
        with self.to_proxy_socket() as socket:
            # Ping RPC Server with request.
            await socket.send_multipart([cloudpickle.dumps(request)])

            # Await acknowledgement from RPCServer.
            if timeout is not None and await socket.poll(timeout=timeout) == 0:
                raise TimeoutError(f"server didn't reply within {timeout} ms")

            response = cloudpickle.loads(await socket.recv())

        if not isinstance(response, str) or response != VLLM_RPC_SUCCESS_STR:
            # Propogate Exception.
            if isinstance(response, Exception):
                logger.warning(error_message)
                raise response
            raise ValueError(error_message)

        return response

    async def get_tokenizer(self, lora_request: LoRARequest):
        return await self.tokenizer.get_lora_tokenizer_async(lora_request)

    async def get_decoding_config(self) -> DecodingConfig:
        return self.decoding_config

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def is_tracing_enabled(self) -> bool:
        return self.tracing_flag

    async def _wait_for_server_rpc(self):
        """Wait for the RPCServer to start up."""

        await self._send_one_way_rpc_request(
            request=RPCUtilityRequest.IS_SERVER_READY,
            error_message="Unable to start RPC Server.",
            timeout=VLLM_RPC_HEALTH_TIMEOUT_MS)

    async def _check_health_rpc(self) -> None:
        """Raise if unhealthy"""

        await self._send_one_way_rpc_request(
            request=RPCUtilityRequest.IS_SERVER_HEALTHY,
            error_message="Did not get HEALTHY response from RPC Server",
            timeout=VLLM_RPC_HEALTH_TIMEOUT_MS)
        
    async def _get_model_config_rpc(self) -> ModelConfig:
        """Get the ModelConfig object from the RPC Server"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_MODEL_CONFIG,
            expected_type=ModelConfig,
            error_message="Could not get ModelConfig from RPC Server")

    async def _get_decoding_config_rpc(self) -> DecodingConfig:
        """Get DecodingConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_DECODING_CONFIG,
            expected_type=DecodingConfig,
            error_message="Could not get DecodingConfig from RPC Server")

    async def _get_parallel_config_rpc(self) -> ParallelConfig:
        """Get ParallelConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_PARALLEL_CONFIG,
            expected_type=ParallelConfig,
            error_message="Could not get ParallelConfig from RPC Server")

    async def _get_scheduler_config_rpc(self) -> SchedulerConfig:
        """Get SchedulerConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_SCHEDULER_CONFIG,
            expected_type=SchedulerConfig,
            error_message="Could not get SchedulerConfig from RPC Server")

    async def _get_lora_config_rpc(self) -> LoRAConfig:
        """Get LoRAConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_LORA_CONFIG,
            expected_type=LoRAConfig,
            error_message="Could not get LoRAConfig from RPC Server")

    async def _is_tracing_enabled_rpc(self) -> bool:
        """Get is_tracing_enabled flag from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.IS_TRACING_ENABLED,
            expected_type=bool,
            error_message="Could not get is_tracing_enabled flag from RPC "
            "Server")

    async def abort(self, request_id: str):
        """Send an ABORT_REQUEST signal to the RPC Server"""

        await self._send_one_way_rpc_request(
            request=RPCAbortRequest(request_id),
            error_message=f"RPCAbortRequest {request_id} failed")

    async def do_log_stats(self):
        """Send a DO_LOG_STATS signal to the RPC Server"""

        await self._send_one_way_rpc_request(
            request=RPCUtilityRequest.DO_LOG_STATS,
            error_message="RPCRequest DO_LOG_STATS failed.")

    async def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> AsyncGenerator[RequestOutput, None]:
        """Send an RPCGenerateRequest to the RPCServer and stream responses."""

        finished = False
        try:
            with self.to_proxy_socket() as socket:

                # Send RPCGenerateRequest.
                await socket.send_multipart([
                    cloudpickle.dumps(
                        RPCGenerateRequest(
                            inputs=inputs,
                            sampling_params=sampling_params,
                            request_id=request_id,
                            lora_request=lora_request,
                            trace_headers=trace_headers,
                            prompt_adapter_request=prompt_adapter_request))
                ])

                # Stream back the results.
                while not finished:
                    message = await socket.recv()
                    request_output = cloudpickle.loads(message)

                    if isinstance(request_output, Exception):
                        # On exception, check if the server is still healthy.
                        # Use this to set the sync `is_running` and `errored`
                        # properties.
                        try:
                            await self._check_health_rpc()
                        except Exception:
                            self._errored = True
                        # NB: do before raising here so that the flag is set
                        # by the time the caller receives this exception
                        raise request_output

                    finished = request_output.finished
                    yield request_output
        finally:
            if not finished:
                await self.abort(request_id)


    async def encode(self, *args,
                     **kwargs) -> AsyncGenerator[EmbeddingRequestOutput, None]:
        raise NotImplementedError(
            "Embeddings not supported with multiprocessing backend")
