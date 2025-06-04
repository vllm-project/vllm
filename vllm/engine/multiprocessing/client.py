# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import copy
import pickle
from contextlib import contextmanager, suppress
from typing import (Any, AsyncGenerator, Dict, Iterator, List, Mapping,
                    Optional, Union, cast)

import cloudpickle
import psutil
import zmq
import zmq.asyncio
from zmq import Frame  # type: ignore[attr-defined]
from zmq.asyncio import Socket

from vllm import PoolingParams
from vllm.config import DecodingConfig, ModelConfig, VllmConfig
from vllm.core.scheduler import SchedulerOutputs
# yapf conflicts with isort for this block
# yapf: disable
from vllm.engine.async_llm_engine import (
    build_guided_decoding_logits_processor_async)
from vllm.engine.multiprocessing import (ENGINE_DEAD_ERROR, IPC_DATA_EXT,
                                         IPC_HEALTH_EXT, IPC_INPUT_EXT,
                                         IPC_OUTPUT_EXT, RPC_REQUEST_T,
                                         VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCAdapterLoadedResponse, RPCError,
                                         RPCIsSleepingRequest,
                                         RPCIsSleepingResponse,
                                         RPCLoadAdapterRequest,
                                         RPCProcessRequest,
                                         RPCResetMultiModalCacheRequest,
                                         RPCResetPrefixCacheRequest,
                                         RPCSleepRequest, RPCStartupRequest,
                                         RPCStartupResponse,
                                         RPCUProfileRequest, RPCWakeUpRequest)
from vllm.engine.protocol import EngineClient
# yapf: enable
from vllm.envs import VLLM_RPC_TIMEOUT
from vllm.inputs import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import Device

logger = init_logger(__name__)


class MQClientClosedError(Exception):
    """Exception class raised when the client is used post-close.

    The client can be closed, which closes the ZMQ context. This normally
    happens on server shutdown. In some cases, methods like abort and
    do_log_stats will still be called and then try to open a socket, which
    causes a ZMQError and creates a huge stack trace.
    So, we throw this error such that we can suppress it.
    """


class MQLLMEngineClient(EngineClient):
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

    def __init__(self, ipc_path: str, engine_config: VllmConfig,
                 engine_pid: int):
        self.context = zmq.asyncio.Context()
        self._errored_with: Optional[BaseException] = None

        # Get the configs.
        self.vllm_config = engine_config
        self.model_config = engine_config.model_config
        self.decoding_config = engine_config.decoding_config

        # Create the tokenizer group.
        self.tokenizer = init_tokenizer_from_configs(
            model_config=self.model_config,
            scheduler_config=engine_config.scheduler_config,
            lora_config=engine_config.lora_config)
        self.input_preprocessor = InputPreprocessor(self.model_config,
                                                    self.tokenizer)

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

        # Loop to handle output of the LLMEngine periodically.
        # Started after the MQLLMEngine is ready so that we can
        # build the Client in an executor to enable clean shutdown.
        self.output_loop: Optional[asyncio.Task] = None

        # Loop to check health of the LLMEngine periodically.
        # Started after the MQLLMEngine is ready.
        self.health_loop: Optional[asyncio.Task] = None
        self._engine_process = psutil.Process(engine_pid)

    @staticmethod
    def is_unsupported_config(vllm_config: VllmConfig):
        # Pipeline parallel not yet supported
        return vllm_config.parallel_config.pipeline_parallel_size > 1

    @contextmanager
    def get_data_socket(self) -> Iterator[Socket]:
        socket = self.context.socket(zmq.constants.DEALER)
        try:
            socket.connect(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    async def run_heartbeat_loop(self, timeout: int):
        """Background loop that continually checks to ensure the engine process
        is still alive.
        """
        try:
            while True:
                # Check if the engine process is running:
                if not self._engine_process.is_running() or (
                        self._engine_process.status() == psutil.STATUS_ZOMBIE):
                    # NB: is_running() returns True for zombies
                    self._set_errored(
                        RuntimeError(
                            f"Engine process (pid {self._engine_process.pid}) "
                            "died."))
                    break

                if await self.heartbeat_socket.poll(timeout=timeout):
                    # Heartbeat received- check the message
                    await self._check_success(
                        error_message="Heartbeat failed.",
                        socket=self.heartbeat_socket)

                logger.debug("Heartbeat successful.")

        except asyncio.CancelledError:
            logger.debug("Shutting down MQLLMEngineClient check health loop.")

        except psutil.NoSuchProcess:
            self._set_errored(
                RuntimeError(
                    f"Engine process (pid {self._engine_process.pid}) died."))

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
                        # If engine is errored, no matter the type of exception
                        # it will no longer be able to receive new requests,
                        # therefore we have to inform that the current
                        # processed requests failed as well. Send back a dead
                        # engine error give this feedback and also give a
                        # 'hint' to the server to shutdown next.
                        exception = self.dead_error

                    if request_id is None:
                        # If request_id is None, then the engine raised an
                        # exception for a batch, and we may not know the
                        # request that caused it, neither if it was actually
                        # caused by any of them (e.g. CUDA OOM). Therefore we
                        # broadcast the same exception for all requests.
                        for queue_i in tuple(self.output_queues.values()):
                            queue_i.put_nowait(exception)
                    else:
                        queue = self.output_queues.get(request_id)
                        if queue is not None:
                            queue.put_nowait(exception)
                # Put each output into the appropriate queue.
                elif isinstance(
                        request_outputs,
                    (RPCAdapterLoadedResponse, RPCIsSleepingResponse)):
                    self._add_output(request_outputs)
                else:
                    for request_output in request_outputs:
                        self._add_output(request_output)

        except asyncio.CancelledError:
            logger.debug("Shutting down MQLLMEngineClient output handler.")

    def _add_output(self, request_output: Union[RequestOutput,
                                                RPCAdapterLoadedResponse,
                                                RPCIsSleepingResponse]):
        queue = self.output_queues.get(request_output.request_id)
        if queue is not None:
            queue.put_nowait(request_output)

    async def setup(self):
        """Setup the client before it starts sending server requests."""

        # Start output_loop
        if self.output_loop is None:
            # only generate once to avoid multiple concurrent output_loops
            # this will lead to race conditions and wrong orders of tokens
            # returned by the engine
            # setup will be called multiple times during the startup of
            # the engine
            self.output_loop = asyncio.create_task(
                self.run_output_handler_loop())

        with self.get_data_socket() as socket:
            # Wait until server is ready.
            response = await self._wait_for_server_rpc(socket)

            self.tracing_flag = response.tracing_enabled

            # Start health_loop.
            if self.health_loop is None:
                self.health_loop = asyncio.create_task(
                    self.run_heartbeat_loop(timeout=VLLM_RPC_TIMEOUT))

    def close(self):
        """Destroy the ZeroMQ Context."""
        # Close all sockets and terminate the context.
        self.context.destroy(linger=0)

        # Cancel background tasks.
        if self.health_loop is not None:
            self.health_loop.cancel()
        if self.output_loop is not None:
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

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return self.input_preprocessor

    async def get_tokenizer(self, lora_request: Optional[LoRARequest] = None):
        return await self.tokenizer.get_lora_tokenizer_async(lora_request)

    async def get_vllm_config(self) -> VllmConfig:
        return self.vllm_config

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

    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[List[SamplerOutput]] = None,
    ) -> None:
        """
        Ignore do_log_stats (handled on MQLLMEngine polling)
        """
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
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt to the LLM. See
                [`PromptType`][vllm.inputs.PromptType] for more details about
                the format of each input.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.
            prompt_adapter_request: Prompt Adapter request to use
                                            for generation, if any.
            priority: Priority of the request (lower means earlier handling).
                Any priority other than 0 will lead to an error if the
                scheduling policy is not "priority".
        """
        return cast(
            AsyncGenerator[RequestOutput, None],
            self._process_request(prompt, sampling_params, request_id,
                                  lora_request, trace_headers,
                                  prompt_adapter_request, priority))

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt to the LLM. See
                [`PromptType`][vllm.inputs.PromptType] for more details about
                the format of each input.
            pooling_params: The pooling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.

        Yields:
            The output `PoolingRequestOutput` objects from the LLMEngine
            for the request.
        """
        return cast(
            AsyncGenerator[PoolingRequestOutput, None],
            self._process_request(prompt,
                                  pooling_params,
                                  request_id,
                                  lora_request,
                                  trace_headers,
                                  priority=priority))

    async def _process_request(
        self,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> Union[AsyncGenerator[RequestOutput, None], AsyncGenerator[
            PoolingRequestOutput, None]]:
        """Send an RPCGenerateRequest to the RPCServer and stream responses."""

        # If already dead, error out.
        if self._errored_with is not None:
            raise ENGINE_DEAD_ERROR(self._errored_with)

        # Ensure the request id is unique among running requests
        if request_id in self.output_queues:
            raise ValueError(f"Request {request_id} already exists")

        # Constructing guided decoding logits processors is expensive, so we do
        # it here to avoid contending with cpu resources and the GIL on the
        # backend process.
        if isinstance(params, SamplingParams) and \
            params.guided_decoding is not None:
            params = await \
                build_guided_decoding_logits_processor_async(
                    sampling_params=params,
                    tokenizer=await self.get_tokenizer(lora_request),
                    default_guided_backend=(self.decoding_config.backend
                        if self.decoding_config
                        else DecodingConfig.backend),
                    model_config=self.model_config,
                    reasoning_backend=self.decoding_config.reasoning_backend,
                )

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
                    prompt=prompt,
                    params=params,
                    request_id=request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=priority,
                ))

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

    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""

        await self._send_one_way_rpc_request(
            request=RPCResetMultiModalCacheRequest.RESET,
            socket=self.input_socket)

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        """Reset the prefix cache"""

        await self._send_one_way_rpc_request(
            request=RPCResetPrefixCacheRequest(device),
            socket=self.input_socket)

    async def sleep(self, level: int = 1) -> None:
        """Sleep the engine for a given level"""
        return await self._send_one_way_rpc_request(
            request=RPCSleepRequest(level), socket=self.input_socket)

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        """Wake up the engine"""
        return await self._send_one_way_rpc_request(
            request=RPCWakeUpRequest(tags), socket=self.input_socket)

    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        request = RPCIsSleepingRequest()

        queue: asyncio.Queue[Union[BaseException,
                                   RPCIsSleepingResponse]] = asyncio.Queue()
        self.output_queues[request.request_id] = queue

        request_bytes = pickle.dumps(request)
        await self.input_socket.send_multipart((request_bytes, ), copy=False)

        request_output = await queue.get()
        self.output_queues.pop(request.request_id)

        if isinstance(request_output, BaseException):
            raise request_output
        return request_output.is_sleeping

    async def add_lora(self, lora_request: LoRARequest) -> None:
        """Load a new LoRA adapter into the engine for future requests."""
        # Uses the same I/O as generate requests
        request = RPCLoadAdapterRequest(lora_request)

        # Create output queue for this requests.
        queue: asyncio.Queue[Union[None, BaseException]] = asyncio.Queue()
        self.output_queues[request.request_id] = queue

        # Send the request
        request_bytes = pickle.dumps(request)
        await self.input_socket.send_multipart((request_bytes, ), copy=False)

        # Wait for the response
        request_output = await queue.get()
        self.output_queues.pop(request.request_id)

        # Raise on error, otherwise happily return None
        if isinstance(request_output, BaseException):
            raise request_output
