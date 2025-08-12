import asyncio
import copy
import pickle
from contextlib import contextmanager, suppress
from typing import (Any, AsyncGenerator, Dict, Iterator, Mapping, Optional,
                    Union)

import cloudpickle
import zmq
import zmq.asyncio
from zmq import Frame  # type: ignore[attr-defined]
from zmq.asyncio import Socket

from vllm import PoolingParams
from vllm.config import DecodingConfig, EngineConfig, ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
# yapf conflicts with isort for this block
# yapf: disable
from vllm.engine.multiprocessing import (ENGINE_DEAD_ERROR, IPC_DATA_EXT,
                                         IPC_HEALTH_EXT, IPC_INPUT_EXT,
                                         IPC_OUTPUT_EXT, RPC_REQUEST_T,
                                         VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCError, RPCProcessRequest,
                                         RPCStartupRequest, RPCStartupResponse,
                                         RPCUProfileRequest)
# yapf: enable
from vllm.envs import VLLM_RPC_TIMEOUT
from vllm.inputs import PromptInputs
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

logger = init_logger(__name__)


class MQClientClosedError(Exception):
    """Exception class raised when the client is used post-close.

    The client can be closed, which closes the ZMQ context. This normally
    happens on server shutdown. In some cases, methods like abort and
    do_log_stats will still be called and then try to open a socket, which
    causes a ZMQError and creates a huge stack trace.
    So, we throw this error such that we can suppress it.
    """


class MQLLMEngineClient:
    """A client wrapper for MQLLMEngine that conforms to the
    EngineClient protocol.

    MQLLMEngine and MQLLMEngineClient are intended to run in separate
    processes communicating via zeromq ipc sockets.

    The entrypoint to MQLLMEngineClient is through the generate()
    method. On generate() MQLLMEngine does three things:
        - Creates an asyncio output queue
        - Sends a RPCGenerateRequest to the MQLLMEngine via zmq
        - Pulls RequestOutputs from its queue and yields them

    MQLLMEngine runs two background loops:
        - output_loop: the output loop pulls List[RequestOutput]
            from the MQLLMEngine via zmq (each list is the output
            of one engine_step in the LLMEngine). It then parses
            the list and pushes individual request_outputs into
            the corresponding output_queue such that they can be
            consumed by the .generate() method.
        - health_loop: the health loop queries the health socket
            every N seconds, confirming the engine is healthy
    """

    def __init__(self, ipc_path: str, engine_config: EngineConfig):
        self.context = zmq.asyncio.Context()
        self._errored_with: Optional[BaseException] = None

        # Get the configs.
        self.model_config = engine_config.model_config
        self.decoding_config = engine_config.decoding_config

        # Create the tokenizer group.
        self.tokenizer = init_tokenizer_from_configs(
            model_config=self.model_config,
            scheduler_config=engine_config.scheduler_config,
            parallel_config=engine_config.parallel_config,
            enable_lora=bool(engine_config.lora_config),
        )

        # Send RPCGenerateRequest to the MQLLMEngine.
        self.input_socket: Socket = self.context.socket(zmq.constants.PUSH)
        self.input_socket.connect(f"{ipc_path}{IPC_INPUT_EXT}")

        # Receive streams of RequestOutput from the MQLLMEngine.
        self.output_socket: Socket = self.context.socket(zmq.constants.PULL)
        self.output_socket.connect(f"{ipc_path}{IPC_OUTPUT_EXT}")

        # IPC path for acking heartbeats.
        self.heartbeat_socket: Socket = self.context.socket(zmq.constants.PULL)
        self.heartbeat_socket.connect(f"{ipc_path}{IPC_HEALTH_EXT}")

        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"

        # Stream for each individual request.
        self.output_queues: Dict[str, asyncio.Queue] = {}
        self.output_loop = asyncio.create_task(self.run_output_handler_loop())

        # Loop to check health of the LLMEngine periodically.
        # Started after the MQLLMEngine is ready.
        self.health_loop: Optional[asyncio.Task] = None

    @staticmethod
    def is_unsupported_config(engine_args: AsyncEngineArgs):
        # Pipeline parallel not yet supported
        return engine_args.pipeline_parallel_size > 1

    @contextmanager
    def get_data_socket(self) -> Iterator[Socket]:
        socket = self.context.socket(zmq.constants.DEALER)
        try:
            socket.connect(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    async def run_heartbeat_loop(self, timeout: int):
        """Background loop that continually listens to the RPCServer for
        heartbeats.
        """
        try:
            while True:
                if await self.heartbeat_socket.poll(timeout=timeout) == 0:
                    # No heartbeat was received. Set error and exit the loop
                    self._set_errored(
                        TimeoutError("No heartbeat received "
                                     "from MQLLMEngine"))
                    logger.debug("Shutting down MQLLMEngineClient check "
                                 "health loop due to timeout")
                    break

                else:
                    # Heartbeat received- check the message
                    await self._check_success(
                        error_message="Heartbeat failed.",
                        socket=self.heartbeat_socket)

                logger.debug("Heartbeat successful.")

        except asyncio.CancelledError:
            logger.debug("Shutting down MQLLMEngineClient check health loop.")

        except Exception as e:
            self._set_errored(e)

    async def run_output_handler_loop(self):
        """Get RequestOutputs from Engine and stream to Request Queues"""

        try:
            while True:
                # Poll, checking for ENGINE_DEAD
                while await self.output_socket.poll(timeout=VLLM_RPC_TIMEOUT
                                                    ) == 0:
                    logger.debug("Waiting for output from MQLLMEngine.")

                    # If errored, alert all running requests.
                    if self.errored:
                        for queue_j in tuple(self.output_queues.values()):
                            queue_j.put_nowait(
                                ENGINE_DEAD_ERROR(self._errored_with))
                        return

                message: Frame = await self.output_socket.recv(copy=False)
                request_outputs = pickle.loads(message.buffer)

                is_error = isinstance(request_outputs,
                                      (BaseException, RPCError))
                if is_error:
                    if isinstance(request_outputs, RPCError):
                        rpc_error: RPCError = request_outputs
                        request_id = rpc_error.request_id
                        exception = rpc_error.exception
                        is_engine_errored = rpc_error.is_engine_errored
                    else:
                        # MPLLMEngine should always return an RPCError to
                        # the output_socket when an issue arises.
                        # If we are here, we are in a bad state and
                        # should shut down the server.
                        error: BaseException = request_outputs
                        logger.error(
                            "Received Exception %s rather than RPCError from "
                            "MPLLMEngine. This should never happen.", error)
                        request_id = None
                        exception = error
                        is_engine_errored = True

                    # Set to error state only on engine critical error
                    # (and record only the first one)
                    if is_engine_errored and not self._errored_with:
                        self._errored_with = exception

                    if request_id is None:
                        for queue_i in tuple(self.output_queues.values()):
                            queue_i.put_nowait(exception)
                    else:
                        queue = self.output_queues.get(request_id)
                        if queue is not None:
                            queue.put_nowait(exception)
                else:
                    # Put each output into the appropriate steam.
                    for request_output in request_outputs:
                        queue = self.output_queues.get(
                            request_output.request_id)
                        if queue is not None:
                            queue.put_nowait(request_output)

        except asyncio.CancelledError:
            logger.debug("Shutting down MQLLMEngineClient output handler.")

    async def setup(self):
        """Setup the client before it starts sending server requests."""

        with self.get_data_socket() as socket:
            # Wait until server is ready.
            response = await self._wait_for_server_rpc(socket)

            self.tracing_flag = response.tracing_enabled

            # Start health_loop.
            self.health_loop = asyncio.create_task(
                self.run_heartbeat_loop(timeout=VLLM_RPC_TIMEOUT))

    def close(self):
        """Destroy the ZeroMQ Context."""
        # Close all sockets and terminate the context.
        self.context.destroy(linger=0)

        # Cancel background tasks.
        if self.health_loop is not None:
            self.health_loop.cancel()
        self.output_loop.cancel()

    def _set_errored(self, e: BaseException):
        logger.exception(repr(e))
        if self._errored_with is None:
            self._errored_with = e

    @staticmethod
    async def _send_get_data_rpc_request(request: RPCStartupRequest,
                                         expected_type: Any,
                                         error_message: str,
                                         socket: Socket) -> Any:
        """Send an RPC request that is expecting data back."""

        # Ping RPCServer with a request.
        await socket.send_multipart((pickle.dumps(request), ), copy=False)

        # Make sure the server responds in time.
        if await socket.poll(timeout=VLLM_RPC_TIMEOUT) == 0:
            raise TimeoutError("RPCServer didn't reply within "
                               f"{VLLM_RPC_TIMEOUT} ms")

        # Await the data from the Server.
        frame = await socket.recv(copy=False)
        data = pickle.loads(frame.buffer)

        if isinstance(data, BaseException):
            raise data
        elif not isinstance(data, expected_type):
            raise ValueError(error_message)

        return data

    @staticmethod
    async def _send_one_way_rpc_request(request: RPC_REQUEST_T,
                                        socket: Socket):
        """Send one-way RPC request to trigger an action."""

        if socket.closed:
            raise MQClientClosedError()

        await socket.send_multipart((pickle.dumps(request), ))

    async def _await_ack(self, error_message: str, socket: Socket):
        """Await acknowledgement that a request succeeded."""

        if socket.closed:
            raise MQClientClosedError()

        if await socket.poll(timeout=VLLM_RPC_TIMEOUT) == 0:
            raise TimeoutError("MQLLMEngine didn't reply within "
                               f"{VLLM_RPC_TIMEOUT}ms")

        await self._check_success(error_message, socket)

    @staticmethod
    async def _check_success(error_message: str, socket: Socket):
        """Confirm that socket has a VLLM_RPC_SUCCESS_STR message"""

        if socket.closed:
            raise MQClientClosedError()

        frame = await socket.recv(copy=False)
        response = pickle.loads(frame.buffer)

        # Raise error if unsuccessful
        if isinstance(response, BaseException):
            raise response
        elif (not isinstance(response, str)
              or response != VLLM_RPC_SUCCESS_STR):
            raise ValueError(error_message)

    async def get_tokenizer(self, lora_request: LoRARequest):
        return await self.tokenizer.get_lora_tokenizer_async(lora_request)

    async def get_decoding_config(self) -> DecodingConfig:
        return self.decoding_config

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def is_tracing_enabled(self) -> bool:
        return self.tracing_flag

    async def _wait_for_server_rpc(self, socket: Socket) -> RPCStartupResponse:
        """Wait for the RPCServer to start up."""

        return await self._send_get_data_rpc_request(
            request=RPCStartupRequest.IS_SERVER_READY,
            expected_type=RPCStartupResponse,
            error_message="Unable to start RPC Server",
            socket=socket)

    async def abort(self, request_id: str):
        """Send an ABORT_REQUEST signal to the RPC Server"""

        with suppress(MQClientClosedError):
            await self._send_one_way_rpc_request(
                request=RPCAbortRequest(request_id), socket=self.input_socket)

    async def do_log_stats(self):
        """Ignore do_log_stats (handled on MQLLMEngine polling)"""
        pass

    async def check_health(self):
        """
        The check health loop probes the health status of the
        Engine's health every N seconds and sets _errored_with
        if the engine is unhealthy.
        """
        if self._errored_with is not None:
            raise self._errored_with

    @property
    def is_running(self) -> bool:
        return not self.errored

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    @property
    def dead_error(self) -> BaseException:
        return ENGINE_DEAD_ERROR(self._errored_with)

    def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            inputs: The inputs to the LLM. See
                :class:`~vllm.inputs.PromptInputs`
                for more details about the format of each input.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.
            prompt_adapter_request: Prompt Adapter request to use
                                            for generation, if any.
        """
        return self._process_request(inputs, sampling_params, request_id,
                                     lora_request, trace_headers,
                                     prompt_adapter_request)

    def encode(
        self,
        inputs: PromptInputs,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
    ) -> AsyncGenerator[EmbeddingRequestOutput, None]:
        """Generate outputs for a request from an embedding model.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            inputs: The inputs to the LLM. See
                :class:`~vllm.inputs.PromptInputs`
                for more details about the format of each input.
            pooling_params: The pooling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.

        Yields:
            The output `EmbeddingRequestOutput` objects from the LLMEngine
            for the request.
        """
        return self._process_request(inputs, pooling_params, request_id,
                                     lora_request, trace_headers)

    async def _process_request(
        self,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> Union[AsyncGenerator[RequestOutput, None], AsyncGenerator[
            EmbeddingRequestOutput, None]]:
        """Send an RPCGenerateRequest to the RPCServer and stream responses."""

        # If already dead, error out.
        if self._errored_with is not None:
            raise ENGINE_DEAD_ERROR(self._errored_with)

        # 1) Create output queue for this requests.
        queue: asyncio.Queue[Union[RequestOutput,
                                   BaseException]] = asyncio.Queue()
        self.output_queues[request_id] = queue

        try:
            # 2) Detach logits processors so that they can be pickled
            # separately (may require cloudpickle which is slower)
            if isinstance(params, SamplingParams) and params.logits_processors:
                # Defensive shallow copy
                params = copy.copy(params)
                logits_processors = params.logits_processors
                params.logits_processors = None
                lp_bytes = cloudpickle.dumps(logits_processors)
            else:
                lp_bytes = None

            request_bytes = pickle.dumps(
                RPCProcessRequest(
                    inputs=inputs,
                    params=params,
                    request_id=request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request))

            # 3) Send the RPCGenerateRequest to the MQLLMEngine.
            parts = (request_bytes,
                     lp_bytes) if lp_bytes else (request_bytes, )
            await self.input_socket.send_multipart(parts, copy=False)

            # 4) Stream the RequestOutputs from the output queue. Note
            # that the output_loop pushes RequestOutput objects to this
            # queue after pulling them from the zmq socket.
            finished = False
            try:
                while not finished:
                    request_output = await queue.get()

                    if isinstance(request_output, BaseException):
                        raise request_output

                    finished = request_output.finished
                    yield request_output
            finally:
                # Request was canceled by the client.
                if not finished and not self.errored:
                    await self.abort(request_id)
        finally:
            self.output_queues.pop(request_id)

    async def start_profile(self) -> None:
        """Start profiling the engine"""

        await self._send_one_way_rpc_request(
            request=RPCUProfileRequest.START_PROFILE, socket=self.input_socket)

    async def stop_profile(self) -> None:
        """Stop profiling the engine"""

        await self._send_one_way_rpc_request(
            request=RPCUProfileRequest.STOP_PROFILE, socket=self.input_socket)
