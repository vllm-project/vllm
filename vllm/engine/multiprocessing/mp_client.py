import asyncio
import pickle
from contextlib import contextmanager, suppress
from typing import (Any, AsyncGenerator, Dict, Iterator, Mapping, Optional,
                    Union)
from uuid import uuid4

import cloudpickle
import zmq
import zmq.asyncio
from zmq import Frame  # type: ignore[attr-defined]
from zmq.asyncio import Socket

from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
# yapf: disable
from vllm.engine.multiprocessing import (RPC_REQUEST_TYPE,
                                         VLLM_RPC_SUCCESS_STR,
                                         RPCAbortRequest,
                                         RPCGenerateRequest,
                                         RPCUtilityRequest)
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


class MPEngineClient:

    def __init__(self, ipc_path: str):
        self.context = zmq.asyncio.Context()
        self._errored = False

        # Send RPCGenerateRequest to the MPLLMEngine.
        self.input_socket: Socket = self.context.socket(zmq.constants.PUSH)
        self.input_socket.connect(f"{ipc_path}_input_socket")

        # Recieve streams of RequestOutput from the MPLLMEngine.
        self.output_socket: Socket = self.context.socket(zmq.constants.PULL)
        self.output_socket.connect(f"{ipc_path}_output_socket")

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

    async def run_output_handler(self):
        # Stream lists of RequestOutput from MPLLMEngine.
        while True:
            message: Frame = await self.output_socket.recv(copy=False)
            request_outputs = pickle.loads(message.buffer)

            for output in request_outputs:
                if isinstance(output, tuple):
                    # Exception case
                    request_id, output = output
                else:
                    request_id = output.request_id

                queue = self.output_queues.get(request_id)
                if queue is not None:
                    queue.put_nowait(output)

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

        await self._notify_ready()

    def close(self):
        """Destroy the ZeroMQ Context."""
        # Close all sockets associated with this context and
        # then terminate the context.
        self.context.destroy(linger=0)


    async def _send_get_data_rpc_request(self, request: RPCUtilityRequest,
                                         expected_type: Any,
                                         error_message: str) -> Any:
        """Send an RPC request that is expecting data back."""

        with self.get_data_socket() as socket:
            # Ping RPCServer with a request.
            await socket.send_multipart(
                (cloudpickle.dumps(request), ),
                copy=False)

            # Make sure the server responds
            if await socket.poll(timeout=VLLM_RPC_GET_DATA_TIMEOUT_MS) == 0:
                raise TimeoutError("Server didn't reply within "
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

    async def _send_one_way_rpc_request(self,
                                        request: RPC_REQUEST_TYPE,
                                        error_message: str,
                                        socket: Optional[Socket] = None):
        """Send one-way RPC request to trigger an action."""

        async def do_rpc_call(socket: Socket, request: RPC_REQUEST_TYPE):

            await socket.send_multipart((cloudpickle.dumps(request), ))

            if await socket.poll(timeout=VLLM_RPC_GET_DATA_TIMEOUT_MS) == 0:
                raise TimeoutError("Server didn't reply within "
                                   f"{VLLM_RPC_GET_DATA_TIMEOUT_MS} ms")

            frame = await socket.recv(copy=False)
            return pickle.loads(frame.buffer)

        if socket is None:
            with self.get_data_socket() as socket:
                response = await do_rpc_call(socket, request)
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

    async def _notify_ready(self):
        """Get the RPCServer that the RPCClient is ready"""

        await self._send_one_way_rpc_request(
            request=RPCUtilityRequest.CLIENT_IS_READY,
            error_message="Unable to notify RPC Server of client readiness")

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

                # ack: Frame = await socket.recv(copy=False)
                # if len(ack.buffer) != 0:
                #     exception = pickle.loads(ack.buffer)
                #     raise exception

            while not finished:
                request_output = await queue.get()
                if isinstance(request_output, BaseException):
                    finished = True
                    # On exception, check if the server is still healthy
                    # possibly setting the `errored` property.
                    if not self._errored:
                        try:
                            # await self.check_health(socket=socket)
                            pass
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
