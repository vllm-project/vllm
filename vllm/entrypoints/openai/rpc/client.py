import asyncio
import pickle
from contextlib import contextmanager, suppress
from typing import Any, AsyncGenerator, Iterator, Mapping, Optional
from uuid import uuid4

import cloudpickle
import zmq
import zmq.asyncio
from zmq import Frame  # type: ignore[attr-defined]
from zmq.asyncio import Socket

from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
# yapf: disable
from vllm.entrypoints.openai.rpc import (RPC_REQUEST_TYPE,
                                         VLLM_RPC_SOCKET_LIMIT_CUTOFF,
                                         VLLM_RPC_SUCCESS_STR,
                                         VLLM_RPC_ZMQ_HWM, RPCAbortRequest,
                                         RPCGenerateRequest, RPCUtilityRequest)
# yapf: enable
from vllm.envs import VLLM_RPC_GET_DATA_TIMEOUT_MS
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


class RPCClientClosedError(Exception):
    """Exception class raised when the client is used post-close.
    
    The client can be closed, which closes the ZMQ context. This normally
    happens on server shutdown. In some cases, methods like abort and 
    do_log_stats will still be called and then try to open a socket, which 
    causes a ZMQError and creates a huge stack trace.
    So, we throw this error such that we can suppress it.
    """


class AsyncEngineRPCClient:
    """
    RPCClient that connects to the RPCServer wrapping AsyncLLMEngine.
    
    The overall design mirrors the Asynchronous Client Server Pattern
    https://zguide.zeromq.org/docs/chapter3/#The-Asynchronous-Client-Server-Pattern

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
        DEALER <- inproc -> [ ROUTER | DEALER ] <- ipc -> DEALER
    
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

    Note: we run set_hwm(0) on each socket, which sets the HWM to inf,
    which is required to avoid dropping messages under high load. 
    This is generally not advisable. However, since we are in control
    of both sides of the connection + failure on either side is
    catastrophic to the overall system health and memory profiling
    suggests limited memory overhead relative to asyncio, we will 
    proceed for now.

    See https://zguide.zeromq.org/docs/chapter2/#High-Water-Marks 
    for more details on high water marks.
    """

    def __init__(self, rpc_path: str):
        self.context = zmq.asyncio.Context()
        self._data_timeout = VLLM_RPC_GET_DATA_TIMEOUT_MS
        self._errored = False

        # Maximum number of sockets that can be opened (typically 65536).
        # ZMQ_SOCKET_LIMIT (http://api.zeromq.org/4-2:zmq-ctx-get)
        socket_limit = self.context.get(zmq.constants.SOCKET_LIMIT)
        assert isinstance(socket_limit, int)
        if socket_limit < VLLM_RPC_SOCKET_LIMIT_CUTOFF:
            raise ValueError(
                f"Found zmq.constants.SOCKET_LIMIT={socket_limit}, which caps "
                "the number of concurrent requests vLLM can process. Launch "
                "vLLM with --disable-frontend-multiprocessing and open a "
                "GitHub issue so we can investigate.")

        # We only have 1 ipc connection that uses unix sockets, so
        # safe to set MAX_SOCKETS to the zmq SOCKET_LIMIT (i.e. will
        # not run into ulimit issues)
        self.context.set(zmq.constants.MAX_SOCKETS, socket_limit)

        # IPC connection to RPC Server (uses unix sockets).
        self.to_rpc_server: Socket = self.context.socket(zmq.constants.DEALER)
        self.to_rpc_server.set_hwm(VLLM_RPC_ZMQ_HWM)
        self.to_rpc_server.bind(rpc_path)

        # In process proxy to RPC Server (uses memory-based messaging).
        self.from_api_server: Socket = self.context.socket(
            zmq.constants.ROUTER)
        self.from_api_server.set_hwm(VLLM_RPC_ZMQ_HWM)
        self.from_api_server.bind(INPROC_PROXY_PATH)

        # Asyncio background task for the proxy.
        self.proxy_in_task = asyncio.create_task(
            self.run_proxy(self.from_api_server, self.to_rpc_server))
        self.proxy_out_task = asyncio.create_task(
            self.run_proxy(self.to_rpc_server, self.from_api_server))

        # Since we open 1 inproc socket per request, we have a hard cap on
        # the number of requests that can run in vLLM w. frontend
        # mulitprocessing. This value is used uvicorn to launch
        # with --limit-concurrency to return 503 when server is overloaded.
        # We need 2 sockets per request - 2:
        # 1 for generate(), 1 for abort(), do_log_stats(), check_health()
        self.limit_concurrency = socket_limit // 2 - 2

    async def run_proxy(self, socket_from: Socket, socket_to: Socket):
        """Background task that runs a proxy"""
        while True:
            frames = await socket_from.recv_multipart(copy=False)
            await socket_to.send_multipart(frames, copy=False)

    async def setup(self):
        """Setup the client before it starts sending server requests."""

        # Wait until server is ready.
        await self._wait_for_server_rpc()

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

    def close(self):
        """Destroy the ZeroMQ Context."""
        # Close all sockets associated with this context and
        # then terminate the context.
        self.from_api_server.close()
        self.to_rpc_server.close()
        self.context.destroy()

    @contextmanager
    def to_proxy_socket(self) -> Iterator[Socket]:
        # Connect to the RPCServer via the proxy.

        # Raise a sensible error if the client was already closed.
        # This can happen if a server shutdown is triggered but some coroutines
        # are still running requests.
        # There should not be a race condition with this check because we don't
        # yield to the event loop between here and opening the socket.
        if self.context.closed:
            raise RPCClientClosedError("The ZMQ client has already shut down")

        # Note that we use DEALER to enable asynchronous communication
        # to enable streaming.
        socket = self.context.socket(zmq.constants.DEALER)
        socket.set_hwm(VLLM_RPC_ZMQ_HWM)
        try:
            socket.connect(INPROC_PROXY_PATH)
            yield socket
        finally:
            socket.close(linger=0)

    async def _send_get_data_rpc_request(self, request: RPCUtilityRequest,
                                         expected_type: Any,
                                         error_message: str) -> Any:
        """Send an RPC request that is expecting data back."""

        with self.to_proxy_socket() as socket:
            # Ping RPCServer with a request.
            await socket.send_multipart((cloudpickle.dumps(request), ),
                                        copy=False)

            # Make sure the server responds
            if await socket.poll(timeout=self._data_timeout) == 0:
                raise TimeoutError("Server didn't reply within "
                                   f"{self._data_timeout} ms")

            # Await the data from the Server.
            frame = await socket.recv(copy=False)
            assert isinstance(frame, Frame)
            data = pickle.loads(frame.buffer)

        if isinstance(data, Exception):
            # Re-raise exceptions returned by the server
            raise data

        if not isinstance(data, expected_type):
            # LoRAConfig can be None.
            if expected_type == LoRAConfig and data is None:
                pass
            elif isinstance(data, Exception):
                logger.error(error_message)
                raise data
            else:
                raise ValueError(error_message)

        return data

    async def _send_one_way_rpc_request(self,
                                        request: RPC_REQUEST_TYPE,
                                        error_message: str,
                                        socket: Optional[Socket] = None):
        """Send one-way RPC request to trigger an action."""

        async def do_rpc_call(socket: Socket, request: RPC_REQUEST_TYPE):

            await socket.send_multipart((cloudpickle.dumps(request), ))

            if await socket.poll(timeout=self._data_timeout) == 0:
                raise TimeoutError("Server didn't reply within "
                                   f"{self._data_timeout} ms")

            frame = await socket.recv(copy=False)
            assert isinstance(frame, Frame)
            return pickle.loads(frame.buffer)

        # Make a new socket connection.
        if socket is None:
            with self.to_proxy_socket() as socket:
                response = await do_rpc_call(socket, request)

        # Use existing socket connection.
        else:
            response = await do_rpc_call(socket, request)

        if not isinstance(response, str) or response != VLLM_RPC_SUCCESS_STR:
            if isinstance(response, Exception):
                logger.error(error_message)
                raise response
            raise ValueError(error_message)

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
            error_message="Unable to start RPC Server")

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
            error_message="Could not get is_tracing_enabled from RPC Server")

    async def abort(self, request_id: str):
        """Send an ABORT_REQUEST signal to the RPC Server"""

        # Suppress timeouts as well.
        # In cases where the server is busy processing requests and a very
        # large volume of abort requests arrive, it is likely that the server
        # will not be able to ack all of them in time. We have seen this when
        # we abort 20k requests at once while another 2k are processing- many
        # of them time out, but we see the server successfully abort all of the
        # requests.
        # In this case we assume that the server has received or will receive
        # these abort requests, and ignore the timeout. This prevents a massive
        # wall of `TimeoutError` stack traces.
        with suppress(RPCClientClosedError, TimeoutError):
            await self._send_one_way_rpc_request(
                request=RPCAbortRequest(request_id),
                error_message=f"RPCAbortRequest {request_id} failed")

    async def do_log_stats(self):
        """Send a DO_LOG_STATS signal to the RPC Server"""
        with suppress(RPCClientClosedError):
            await self._send_one_way_rpc_request(
                request=RPCUtilityRequest.DO_LOG_STATS,
                error_message="RPCRequest DO_LOG_STATS failed.")

    @property
    def is_running(self) -> bool:
        return not self._errored

    @property
    def is_stopped(self) -> bool:
        return self._errored

    @property
    def errored(self) -> bool:
        return self._errored

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
                # Send RPCGenerateRequest to the RPCServer.
                await socket.send_multipart((cloudpickle.dumps(
                    RPCGenerateRequest(
                        inputs=inputs,
                        sampling_params=sampling_params,
                        request_id=request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        prompt_adapter_request=prompt_adapter_request)), ))

                # Stream back the results from the RPC Server.
                while not finished:
                    message = await socket.recv(copy=False)
                    assert isinstance(message, Frame)
                    request_output = pickle.loads(message.buffer)

                    if isinstance(request_output, Exception):
                        # On exception, check if the server is still healthy
                        # possibly setting the `errored` property.
                        if not self._errored:
                            try:
                                await self.check_health(socket=socket)
                            except Exception as e:
                                self._errored = True
                                logger.exception(repr(e))

                        # NB: do before raising here so that the flag is set
                        # by the time the caller receives this exception
                        raise request_output

                    finished = request_output.finished
                    yield request_output

        finally:
            # Request was canceled by the client.
            if not finished and not self._errored:
                await self.abort(request_id)

    async def check_health(self, socket: Optional[Socket] = None) -> None:
        """Raise if unhealthy"""

        await self._send_one_way_rpc_request(
            request=RPCUtilityRequest.IS_SERVER_HEALTHY,
            error_message="Got Unhealthy response from RPC Server",
            socket=socket)

    async def encode(self, *args,
                     **kwargs) -> AsyncGenerator[EmbeddingRequestOutput, None]:
        raise NotImplementedError(
            "Embeddings not supported with multiprocessing backend")

    async def start_profile(self) -> None:
        """Start profiling the engine"""

        await self._send_one_way_rpc_request(
            request=RPCUtilityRequest.START_PROFILE,
            error_message="RPCRequest START_PROFILE failed.")

    async def stop_profile(self) -> None:
        """Stop profiling the engine"""

        await self._send_one_way_rpc_request(
            request=RPCUtilityRequest.STOP_PROFILE,
            error_message="RPCRequest STOP_PROFILE failed.")
