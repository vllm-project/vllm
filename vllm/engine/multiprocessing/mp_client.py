import asyncio
import pickle
from contextlib import contextmanager, suppress
from typing import (Any, AsyncGenerator, Dict, Iterator, List, Mapping,
                    Optional, Union)

import cloudpickle
import zmq
import zmq.asyncio
from zmq import Frame  # type: ignore[attr-defined]
from zmq.asyncio import Socket

from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.engine.multiprocessing import (RPC_REQUEST_TYPE,
                                         VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCGenerateRequest, RPCStartupRequest,
                                         RPCUtilityRequest)
from vllm.envs import VLLM_RPC_GET_DATA_TIMEOUT_MS
from vllm.inputs import PromptInputs
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

logger = init_logger(__name__)

CHECK_HEALTH_INTERVAL_S = 10.

class MPClientClosedError(Exception):
    """Exception class raised when the client is used post-close.
    
    The client can be closed, which closes the ZMQ context. This normally
    happens on server shutdown. In some cases, methods like abort and 
    do_log_stats will still be called and then try to open a socket, which 
    causes a ZMQError and creates a huge stack trace.
    So, we throw this error such that we can suppress it.
    """


class MPEngineClient:

    def __init__(self, ipc_path: str):
        self.context = zmq.asyncio.Context()
        self._errored = False
        self._errored_with: Optional[BaseException] = None

        # Send RPCGenerateRequest to the MQLLMEngine.
        self.input_socket: Socket = self.context.socket(zmq.constants.PUSH)
        self.input_socket.connect(f"{ipc_path}_input_socket")

        # Receive streams of RequestOutput from the MQLLMEngine.
        self.output_socket: Socket = self.context.socket(zmq.constants.PULL)
        self.output_socket.connect(f"{ipc_path}_output_socket")

        # IPC path for ack of check_health requests.
        self.health_socket: Socket = self.context.socket(zmq.constants.PULL)
        self.health_socket.connect(f"{ipc_path}_health_socket")

        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}_data_socket"

        # Stream for each individual request.
        self.output_queues: Dict[str, asyncio.Queue] = {}
        self.output_handler = asyncio.create_task(self.run_output_handler())

    @contextmanager
    def get_data_socket(self) -> Iterator[Socket]:
        socket = self.context.socket(zmq.constants.DEALER)
        try:
            socket.connect(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    async def run_check_health_loop(self):
        # Check health periodically.
        while True:
            await asyncio.sleep(CHECK_HEALTH_INTERVAL_S)
            try:
                await self._check_health_rpc(self.health_socket)
            except Exception as e:
                self._errored = True
                self._errored_with = e

    async def run_output_handler(self):
        # Stream lists of RequestOutput from MQLLMEngine.
        while True:
            message: Frame = await self.output_socket.recv(copy=False)
            request_outputs: List[RequestOutput] = pickle.loads(message.buffer)

            for output in request_outputs:
                if isinstance(output, tuple):
                    # Exception case
                    request_id, output = output
                else:
                    request_id = output.request_id

                if request_id is not None:
                    queue = self.output_queues.get(request_id)
                    if queue is not None:
                        queue.put_nowait(output)
                else:
                    # request_id None means apply to all active requests.
                    for queue in tuple(self.output_queues.values()):
                        queue.put_nowait(output)

    async def setup(self):
        """Setup the client before it starts sending server requests."""

        with self.get_data_socket() as socket:

            # Wait until server is ready.
            await self._wait_for_server_rpc(socket)

            # Get the configs.
            self.model_config = await self._get_model_config_rpc(socket)
            self.decoding_config = await self._get_decoding_config_rpc(socket)
            self.tracing_flag = await self._is_tracing_enabled_rpc(socket)

            # Create the tokenizer group.
            # TODO: refactor OAI server to avoid needing this info.
            self.tokenizer = init_tokenizer_from_configs(
                model_config=self.model_config,
                scheduler_config=(await
                                  self._get_scheduler_config_rpc(socket)),
                parallel_config=(await self._get_parallel_config_rpc(socket)),
                enable_lora=bool(await self._get_lora_config_rpc(socket)),
            )

            # Notify MQLLMEngine client is ready to start sending requests.
            await self._notify_ready(socket)

    def close(self):
        """Destroy the ZeroMQ Context."""
        # Close all sockets associated with this context and
        # then terminate the context.
        self.output_socket.close()
        self.input_socket.close()
        self.health_socket.close()
        self.context.destroy(linger=0)

        # TODO: cancel the handler task.

    async def _send_get_data_rpc_request(self, request: RPCStartupRequest,
                                         expected_type: Any,
                                         error_message: str,
                                         socket: Socket) -> Any:
        """Send an RPC request that is expecting data back."""

        # Ping RPCServer with a request.
        await socket.send_multipart((cloudpickle.dumps(request), ), copy=False)

        # Make sure the server responds
        if await socket.poll(timeout=VLLM_RPC_GET_DATA_TIMEOUT_MS) == 0:
            raise TimeoutError("RPCServer didn't reply within "
                               f"{VLLM_RPC_GET_DATA_TIMEOUT_MS} ms")

        # Await the data from the Server.
        frame = await socket.recv(copy=False)
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

    async def _send_one_way_rpc_request(
        self, request: RPC_REQUEST_TYPE, 
        socket: Socket, 
        await_ack: bool = False, 
        error_message: str = "RPCRequest Failed."):
        """Send one-way RPC request to trigger an action."""

        await socket.send_multipart((cloudpickle.dumps(request), ))

        if await_ack:
            await self._ack_one_way_rpc_request(
                expected_str=VLLM_RPC_SUCCESS_STR,
                timeout=VLLM_RPC_GET_DATA_TIMEOUT_MS,
                error_message=error_message,
                socket=socket)


    async def _ack_one_way_rpc_request(
        self, timeout: int, expected_str: str, error_message: str, socket: Socket):
        "Await acknoledgement that a request succeeded."

        if await socket.poll(timeout=timeout) == 0:
            raise TimeoutError(f"MQLLMEngine didn't reply within {timeout}ms")

        frame = await socket.recv(copy=False)
        response = pickle.loads(frame.buffer)

        if not isinstance(response, str) or response != expected_str:
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

    async def _wait_for_server_rpc(self, socket: Socket):
        """Wait for the RPCServer to start up."""

        self._send_one_way_rpc_request(
            request=RPCStartupRequest.IS_SERVER_READY,
            socket=socket,
            await_ack=True,
            error_message="Unable to start RPC Server")

    async def _check_health_rpc(self, socket: Socket):
        """Get current health status from the RPCServer"""

        self._send_one_way_rpc_request(
            request=RPCUtilityRequest.CHECK_HEALTH,
            socket=socket,
            await_ack=True,
            error_message="Check health failed.")

    async def _notify_ready(self, socket: Socket):
        """Get the RPCServer that the RPCClient is ready"""

        await self._send_one_way_rpc_request(
            request=RPCStartupRequest.CLIENT_IS_READY,
            socket=socket)

    async def _get_model_config_rpc(self, socket: Socket) -> ModelConfig:
        """Get the ModelConfig object from the RPC Server"""

        return await self._send_get_data_rpc_request(
            RPCStartupRequest.GET_MODEL_CONFIG,
            expected_type=ModelConfig,
            error_message="Could not get ModelConfig from RPC Server",
            socket=socket)

    async def _get_decoding_config_rpc(self, socket: Socket) -> DecodingConfig:
        """Get DecodingConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCStartupRequest.GET_DECODING_CONFIG,
            expected_type=DecodingConfig,
            error_message="Could not get DecodingConfig from RPC Server",
            socket=socket)

    async def _get_parallel_config_rpc(self, socket: Socket) -> ParallelConfig:
        """Get ParallelConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCStartupRequest.GET_PARALLEL_CONFIG,
            expected_type=ParallelConfig,
            error_message="Could not get ParallelConfig from RPC Server",
            socket=socket)

    async def _get_scheduler_config_rpc(self,
                                        socket: Socket) -> SchedulerConfig:
        """Get SchedulerConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCStartupRequest.GET_SCHEDULER_CONFIG,
            expected_type=SchedulerConfig,
            error_message="Could not get SchedulerConfig from RPC Server",
            socket=socket)

    async def _get_lora_config_rpc(self, socket: Socket) -> LoRAConfig:
        """Get LoRAConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCStartupRequest.GET_LORA_CONFIG,
            expected_type=LoRAConfig,
            error_message="Could not get LoRAConfig from RPC Server",
            socket=socket)

    async def _is_tracing_enabled_rpc(self, socket: Socket) -> bool:
        """Get is_tracing_enabled flag from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCStartupRequest.GET_TRACING_ENABLED,
            expected_type=bool,
            error_message="Could not get is_tracing_enabled from RPC Server",
            socket=socket)

    async def abort(self, request_id: str):
        """Send an ABORT_REQUEST signal to the RPC Server"""

        # Suppress timeouts and MPClientClosedError.
        # In cases where the server is busy processing requests and a very
        # large volume of abort requests arrive, it is likely that the server
        # will not be able to ack all of them in time. We have seen this when
        # we abort 20k requests at once while another 2k are processing- many
        # of them time out, but we see the server successfully abort all of the
        # requests.
        # In this case we assume that the server has received or will receive
        # these abort requests, and ignore the timeout. This prevents a massive
        # wall of `TimeoutError` stack traces.
        with suppress(MPClientClosedError, TimeoutError):
            await self._send_one_way_rpc_request(
                request=RPCAbortRequest(request_id), socket=self.input_socket)

    async def do_log_stats(self):
        """Send a DO_LOG_STATS signal to the RPC Server"""
        with suppress(MPClientClosedError):
            await self._send_one_way_rpc_request(
                request=RPCUtilityRequest.DO_LOG_STATS,
                socket=self.input_socket)

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

        queue: asyncio.Queue[Union[RequestOutput,
                                   BaseException]] = asyncio.Queue()
        self.output_queues[request_id] = queue
        finished = False
        try:

            # Send RPCGenerateRequest to the RPCServer.
            await self.input_socket.send_multipart((cloudpickle.dumps(
                RPCGenerateRequest(
                    inputs=inputs,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request)), ))

            while not finished:
                request_output = await queue.get()
                # todo: convert message to include status of whether the engine is errored
                # such that we dont need another RPC call
                if isinstance(request_output, BaseException):
                    finished = True
                    # On exception, check if the server is still healthy
                    # possibly setting the `errored` property.
                    if not self._errored:
                        try:
                            await self.check_health()
                        except Exception as e:
                            self._errored = True
                            logger.exception(repr(e))
                    raise request_output

                finished = request_output.finished
                yield request_output

        finally:
            self.output_queues.pop(request_id)
            # Request was canceled by the client.
            if not finished and not self._errored:
                await self.abort(request_id)

    async def encode(self, *args,
                     **kwargs) -> AsyncGenerator[EmbeddingRequestOutput, None]:
        raise NotImplementedError(
            "Embeddings not supported with multiprocessing backend")
