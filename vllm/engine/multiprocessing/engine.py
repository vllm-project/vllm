import pickle
from contextlib import contextmanager
from typing import Iterator, List, Union

import cloudpickle
import zmq

from vllm import AsyncEngineArgs, LLMEngine
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.engine.multiprocessing import (ENGINE_DEAD_ERROR, IPC_DATA_EXT,
                                         IPC_HEALTH_EXT, IPC_INPUT_EXT,
                                         IPC_OUTPUT_EXT, REQUEST_OUTPUTS_T,
                                         VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCError, RPCGenerateRequest,
                                         RPCStartupRequest, RPCUtilityRequest)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext

CONFIG_TYPE = Union[ModelConfig, DecodingConfig, ParallelConfig,
                    SchedulerConfig, LoRAConfig]

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 10000
HEALTHY_RESPONSE = (pickle.dumps(VLLM_RPC_SUCCESS_STR), )


class MQLLMEngine:
    """A multiprocessing wrapper for :class:`LLMEngine`.

    This class is used to wrap the :class:`LLMEngine` class to enable use
    in asynchronous manner. It runs a background loop and uses zeromq to 
    receive new requests and stream outputs incrementally to another process.
    
    The :class:`LLMEngine` is kicked off when a new RPCGenerateRequest 
    is received by the input_socket.
    
    The self.engine_loop checks the input_socket for new requests,
    adds them to the LLMEngine if there are any, calls the internal
    :class:`LLMEngine.step()` and sends the RequestOutputs back over
    the output_socket.

    If use_async_sockets is set, the logic associated with reading new
    requests from the socket and sending data to the socket is passed
    as a callback to the llm_engine, which calls the logic asynchronously
    such that the IPC can be overlapped with the GPU.

    Args:
        ipc_path: Base path for zeromq interprocess messaging
        use_async_sockets: Whether to make send/recv async with GPU
        log_requests: Whether to log the requests.
        *args: Arguments for :class:`LLMEngine`.
        **kwargs: Arguments for :class:`LLMEngine`.
    """

    def __init__(self,
                 ipc_path: str,
                 use_async_sockets: bool,
                 *args,
                 log_requests: bool = True,
                 **kwargs) -> None:
        self.engine = LLMEngine(*args, **kwargs)
        self.log_requests = log_requests

        self.use_async_sockets = use_async_sockets
        if self.use_async_sockets:
            self.engine.process_request_outputs_callback = \
                self._async_socket_engine_callback

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # Receive input from the client.
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        # self.input_socket.set_hwm(0)
        self.input_socket.bind(f"{ipc_path}{IPC_INPUT_EXT}")

        # Send output stream back to client.
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        # self.output_socket.set_hwm(0)
        self.output_socket.bind(f"{ipc_path}{IPC_OUTPUT_EXT}")

        # Send health status back to client.
        self.health_socket = self.ctx.socket(zmq.constants.PUSH)
        self.health_socket.bind(f"{ipc_path}{IPC_HEALTH_EXT}")

        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"

        # Error state.
        self._errored = False

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs,
                         usage_context: UsageContext, ipc_path: str):
        """Creates an MQLLMEngine from the engine arguments."""

        engine_config = engine_args.create_engine_config()

        if engine_args.engine_use_ray:
            raise NotImplementedError(
                "--engine-use-ray is not supported for MQLLMEngine. "
                "Launch with --disable-frontend-multiprocessing if you "
                "need to deploy with this flag (not recommended).")

        executor_class = LLMEngine._get_executor_cls(engine_config)

        return cls(
            ipc_path=ipc_path,
            use_async_sockets=engine_config.model_config.use_async_output_proc,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context)

    def start(self):
        try:
            try:
                logger.debug("Starting Startup Loop.")
                self.run_startup_loop()
                logger.debug("Starting Engine Loop.")
                self.run_engine_loop()
            except Exception as e_core:
                try:
                    logger.exception(repr(e_core))
                    if self._errored:
                        logger.debug("Starting Dead Loop.")
                        self.run_engine_dead_loop()
                except Exception as e_dead_loop:
                    logger.exception(repr(e_dead_loop))
        except KeyboardInterrupt:
            logger.debug("Shutting down MQLLMEngine.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup zeromq state on shutdown."""
        self.input_socket.close()
        self.output_socket.close()
        self.health_socket.close()
        self.ctx.destroy(linger=0)
        del self.engine

    @contextmanager
    def make_data_socket(
            self) -> Iterator[zmq.Socket]:  # type: ignore[name-defined]
        socket = self.ctx.socket(zmq.constants.ROUTER)
        try:
            socket.bind(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    def run_startup_loop(self) -> None:
        """Startup loop for sending data from Engine -> Client."""

        with self.make_data_socket() as socket:

            # Loop until the RPCClient has all the data it needs.
            client_is_ready = False
            while not client_is_ready:
                try:
                    identity, message = socket.recv_multipart(copy=False)
                    request: RPCStartupRequest = pickle.loads(message.buffer)

                    # Handle the query from the Client.
                    if request == RPCStartupRequest.GET_MODEL_CONFIG:
                        response = self.engine.get_model_config()
                    elif request == RPCStartupRequest.GET_DECODING_CONFIG:
                        response = self.engine.get_decoding_config()
                    elif request == RPCStartupRequest.GET_LORA_CONFIG:
                        response = self.engine.get_lora_config()
                    elif request == RPCStartupRequest.GET_SCHEDULER_CONFIG:
                        response = self.engine.get_scheduler_config()
                    elif request == RPCStartupRequest.GET_PARALLEL_CONFIG:
                        response = self.engine.get_parallel_config()
                    elif request == RPCStartupRequest.GET_TRACING_ENABLED:
                        response = self.engine.is_tracing_enabled()
                    elif request == RPCStartupRequest.IS_SERVER_READY:
                        response = VLLM_RPC_SUCCESS_STR
                    elif request == RPCStartupRequest.CLIENT_IS_READY:
                        response = VLLM_RPC_SUCCESS_STR
                        # Breakout of loop once client is ready.
                        client_is_ready = True

                    socket.send_multipart((identity, pickle.dumps(response)),
                                          copy=False)

                except Exception as e:
                    socket.send_multipart((identity, pickle.dumps(e)),
                                          copy=False)

    def run_engine_loop(self):
        """Core busy loop of the LLMEngine."""

        while True:
            if not self.engine.has_unfinished_requests():
                # Poll until there is work to do.
                while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                    logger.debug("Waiting for new requests in engine loop.")

            # Handle any input from the client.
            self.handle_new_input()

            # Engine step.
            request_outputs = self.engine_step()

            # Send request outputs (if async, done in engine_step callback).
            if not self.use_async_sockets:
                self._send_outputs(request_outputs)

    def run_engine_dead_loop(self):
        """Loop for replying to all requests that we are dead."""
        if not self._errored:
            raise ValueError("In dead loop, but found _errored=False")

        while True:
            # Poll until there is a request
            while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for new requests in dead loop.")

            # Handle any new data, replying with EngineDeadError
            self.handle_new_input()

    def engine_step(self) -> List[RequestOutput]:
        """Engine step wrapper with error handling."""

        try:
            return self.engine.step()
        except Exception as e:
            self._errored = True
            rpc_err = RPCError(request_id=None,
                               is_engine_errored=True,
                               exception=e)
            logger.exception(repr(e))
            self._send_outputs(rpc_err)
            raise e

    def handle_new_input(self):
        """Handle new input from the socket"""
        try:
            while self.input_socket.poll(timeout=0) != 0:
                message = self.input_socket.recv(copy=False)
                request = cloudpickle.loads(message.buffer)

                if isinstance(request, RPCGenerateRequest):
                    self._handle_generate_request(request)
                elif isinstance(request, RPCAbortRequest):
                    self._handle_abort_request(request)
                elif isinstance(request, RPCUtilityRequest):
                    self._handle_utility_request(request)
                else:
                    raise ValueError("Unknown RPCRequest Type: {request}")

        except Exception as e:
            self._errored = True
            logger.exception(repr(e))
            self._send_unhealthy(e)
            raise e

    def _handle_generate_request(self, request: RPCGenerateRequest):
        """Handle RPCGenerateRequest by adding it to the LLMEngine."""
        request_id = request.request_id

        if self._errored:
            rpc_err = RPCError(request_id=request_id,
                               is_engine_errored=True,
                               exception=ENGINE_DEAD_ERROR)
            self._send_outputs(rpc_err)

        try:
            self.engine.add_request(
                request_id=request_id,
                inputs=request.inputs,
                params=request.sampling_params,
                lora_request=request.lora_request,
                trace_headers=request.trace_headers,
                prompt_adapter_request=request.prompt_adapter_request)

            if self.log_requests:
                logger.info("Added request %s.", request.request_id)

        except Exception as e:
            # We do not set self._errored = True here, since the error
            # is due to an issue adding this request to the engine,
            # rather than an issue with the engine itself.
            rpc_err = RPCError(request_id=request_id,
                               is_engine_errored=self._errored,
                               exception=e)
            self._send_outputs(rpc_err)

            # Remove request from the engine.
            self.engine.abort_request(request_id)

    def _handle_abort_request(self, request: RPCAbortRequest):
        self.engine.abort_request(request.request_id)
        if self.log_requests:
            logger.info("Aborted request %s.", request.request_id)

    def _handle_utility_request(self, request: RPCUtilityRequest):
        if request == RPCUtilityRequest.DO_LOG_STATS:
            if not self._errored:
                self.engine.do_log_stats()
        elif request == RPCUtilityRequest.CHECK_HEALTH:
            if self._errored:
                self._send_unhealthy(ENGINE_DEAD_ERROR)
            try:
                self.engine.check_health()
                self._send_healthy()
            except Exception as e:
                self._send_unhealthy(e)

    def _send_outputs(self, outputs: REQUEST_OUTPUTS_T):
        """Send List of RequestOutput to RPCClient."""
        if outputs:
            output_bytes = pickle.dumps(outputs)
            self.output_socket.send_multipart((output_bytes, ), copy=False)

    def _send_healthy(self):
        """Send HEALTHY message to RPCClient."""
        self.health_socket.send_multipart(HEALTHY_RESPONSE, copy=False)

    def _send_unhealthy(self, error: BaseException):
        """Send UNHEALTHY message to RPCClient."""
        error_bytes = pickle.dumps(error)
        self.health_socket.send_multipart((error_bytes, ), copy=False)

    def _async_socket_engine_callback(self,
                                      request_outputs: REQUEST_OUTPUTS_T):
        """Callback used by engine to make socket handling async with GPU."""
        self._send_outputs(request_outputs)
        self.handle_new_input()


def run_mp_engine(engine_args: AsyncEngineArgs, usage_context: UsageContext,
                  ipc_path: str):
    engine = MQLLMEngine.from_engine_args(engine_args=engine_args,
                                          usage_context=usage_context,
                                          ipc_path=ipc_path)

    engine.start()
