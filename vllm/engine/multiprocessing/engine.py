import pickle
import psutil
import signal
from contextlib import contextmanager
from typing import Iterator, List

import cloudpickle
import zmq

from vllm import AsyncEngineArgs, SamplingParams
from vllm.engine.llm_engine import LLMEngine
# yapf conflicts with isort for this block
# yapf: disable
from vllm.engine.multiprocessing import (REQUEST_OUTPUTS_T,
                                         VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCError, RPCProcessRequest,
                                         RPCUProfileRequest)
# yapf: enable
from vllm.executor.gpu_executor import GPUExecutor
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_exception_traceback, make_zmq_socket

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 10000
HEALTHY_RESPONSE = (pickle.dumps(VLLM_RPC_SUCCESS_STR), )


class MQLLMEngine:
    """A multiprocessing wrapper for :class:`LLMEngine`.

    This class is used to wrap the :class:`LLMEngine` class to enable use
    in concurrnet manner. It runs a background loop and uses zeromq to 
    receive new requests and stream outputs incrementally via ipc.
    
    The :class:`LLMEngine` generate or encode process is kicked off when a new
    RPCProcessRequest is received by the input_socket.
    
    The self.engine_loop checks the input_socket for new requests,
    adds them to the LLMEngine if there are any, calls the internal
    :class:`LLMEngine.step()`, and sends the RequestOutputs back over
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
        # For MQLLMEngine, we can use cached outputs, since each new request
        # output is immediately pickled and send over the socket, which frees
        # the python object to be reused again.
        kwargs['use_cached_outputs'] = True

        self.engine = LLMEngine(*args, **kwargs)
        self.log_requests = log_requests

        self.use_async_sockets = use_async_sockets
        if self.use_async_sockets:
            self.engine.process_request_outputs_callback = \
                self._async_socket_engine_callback

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # Receive input from the client.
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.bind(f"{ipc_path}{IPC_INPUT_EXT}")

        # Send output stream back to client.
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.bind(f"{ipc_path}{IPC_OUTPUT_EXT}")


    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs,
                         usage_context: UsageContext, ipc_path: str):
        """Creates an MQLLMEngine from the engine arguments."""
        # Setup plugins for each process
        from vllm.plugins import load_general_plugins
        load_general_plugins()

        engine_config = engine_args.create_engine_config(usage_context)
        executor_class = LLMEngine._get_executor_cls(engine_config)

        use_async_sockets = engine_config.model_config.use_async_output_proc

        return cls(ipc_path=ipc_path,
                   use_async_sockets=use_async_sockets,
                   vllm_config=engine_config,
                   executor_class=executor_class,
                   log_requests=not engine_args.disable_log_requests,
                   log_stats=not engine_args.disable_log_stats,
                   usage_context=usage_context)

    def cleanup(self):
        """Cleanup zeromq state on shutdown."""
        # Closes all sockets and destroys context.
        self.ctx.destroy(linger=0)
        del self.engine

    def run_engine_loop(self):
        """Core busy loop of the LLMEngine."""

        while True:
            if not self.engine.has_unfinished_requests():
                # Poll until there is work to do.
                while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                    # When there's no work, check on engine health and send
                    # health status back to client
                    self.engine.check_health()
                    self.engine.do_log_stats()
                    logger.debug("Waiting for new requests in engine loop.")

            # Handle any input from the client.
            self.handle_new_input()

            # Engine step.
            request_outputs = self.engine.step()

            # Send request outputs (if async, done in engine_step callback).
            if not self.use_async_sockets:
                self._send_outputs(request_outputs)


    def handle_new_input(self):
        """Handle new input from the socket"""

        while self.input_socket.poll(timeout=0) != 0:
            frames = self.input_socket.recv_multipart(copy=False)
            request = pickle.loads(frames[0].buffer)

            if isinstance(request, RPCProcessRequest):
                if len(frames) > 1:
                    # Use cloudpickle for logits processors
                    assert isinstance(request.params, SamplingParams)
                    lprocs = cloudpickle.loads(frames[1].buffer)
                    request.params.logits_processors = lprocs
                self._handle_process_request(request)
            elif isinstance(request, RPCAbortRequest):
                self._handle_abort_request(request)
            elif isinstance(request, RPCUProfileRequest):
                if request == RPCUProfileRequest.START_PROFILE:
                    self.start_profile()
                else:
                    self.stop_profile()
            else:
                raise ValueError(
                    f"Unknown RPCRequest Type: {type(request)}")

    def _handle_process_request(self, request: RPCProcessRequest):
        """Handle RPCProcessRequest by adding it to the LLMEngine."""
        request_id = request.request_id

        try:
            self.engine.add_request(
                request_id=request_id,
                prompt=request.prompt,
                params=request.params,
                lora_request=request.lora_request,
                trace_headers=request.trace_headers,
                prompt_adapter_request=request.prompt_adapter_request,
                priority=request.priority)

            if self.log_requests:
                logger.info("Added request %s.", request.request_id)

        except Exception as e:
            # We do not set self._errored = True here, since the error
            # is due to an issue adding this request to the engine,
            # rather than an issue with the engine itself.
            rpc_err = RPCError(request_id=request_id,
                               exception=e)
            self._send_outputs(rpc_err)

            # Remove request from the engine.
            self.engine.abort_request(request_id)

    def _handle_abort_request(self, request: RPCAbortRequest):
        self.engine.abort_request(request.request_id)
        if self.log_requests:
            logger.info("Aborted request %s.", request.request_id)

    def _send_outputs(self, outputs: REQUEST_OUTPUTS_T):
        """Send List of RequestOutput to RPCClient."""
        if outputs:
            try:
                from ray.exceptions import RayTaskError

                # RayTaskError might not pickelable here. We need to unpack the
                # underlying exception as the real exception in the output.
                if (isinstance(outputs, RPCError)
                        and isinstance(outputs.exception, RayTaskError)):
                    outputs.exception = outputs.exception.cause
            except ImportError:
                pass

            output_bytes = pickle.dumps(outputs)
            self.output_socket.send_multipart((output_bytes, ), copy=False)

    def _async_socket_engine_callback(self,
                                      request_outputs: REQUEST_OUTPUTS_T):
        """Callback used by engine to make socket handling async with GPU."""
        self._send_outputs(request_outputs)
        self.handle_new_input()

    def start_profile(self) -> None:
        if type(self.engine.model_executor) is GPUExecutor:
            self.engine.model_executor.start_profile()
        else:
            self.engine.model_executor._run_workers("start_profile")

    def stop_profile(self) -> None:
        if type(self.engine.model_executor) is GPUExecutor:
            self.engine.model_executor.stop_profile()
        else:
            self.engine.model_executor._run_workers("stop_profile")
    
    @staticmethod
    def run_mq_llm_engine(
        engine_args: AsyncEngineArgs,
        usage_context: UsageContext,
        ipc_path: str
    ):
        parent_process = psutil.Process().parent()
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            engine = MQLLMEngine.from_engine_args(
                engine_args, usage_context, ipc_path)
            engine.run_engine_loop()
        
        # If an exception arises, log the error and raise a SIGQUIT.
        # The parent process will listen for SIGQUIT and shutdown if
        # it arises. The root cause will show up at the bottom of the
        # stack trace for both startup time and runtime error.
        except Exception:
            traceback = get_exception_traceback()
            logger.error("EngineCore hit an exception: %s", traceback)
            parent_process.send_signal(signal.SIGQUIT)


def signal_handler(*_) -> None:
    raise KeyboardInterrupt("MQLLMEngine terminated")
