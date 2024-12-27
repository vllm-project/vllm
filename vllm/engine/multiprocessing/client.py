import asyncio
import copy
import pickle
from contextlib import suppress
from typing import (AsyncGenerator, Dict, List, Mapping, Optional, Union, cast,
                    overload)

import cloudpickle
import zmq
import zmq.asyncio
from typing_extensions import deprecated
from zmq import Frame  # type: ignore[attr-defined]
from zmq.asyncio import Socket

from vllm import PoolingParams
from vllm.config import DecodingConfig, ModelConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.arg_utils import AsyncEngineArgs
# yapf conflicts with isort for this block
# yapf: disable
from vllm.engine.async_llm_engine import (
    build_guided_decoding_logits_processor_async)
from vllm.engine.multiprocessing import (ENGINE_DEAD_ERROR, RPC_REQUEST_T,
                                         RPCAbortRequest, RPCError,
                                         RPCProcessRequest, RPCUProfileRequest)
from vllm.engine.multiprocessing.engine import MQLLMEngine
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
from vllm.usage.usage_lib import UsageContext
from vllm.utils import deprecate_kwargs, get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.utils import BackgroundProcHandle

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
    """

    def __init__(self,
                 engine_args: AsyncEngineArgs,
                 usage_context: UsageContext = UsageContext.ENGINE_CONTEXT):

        self._errored_with: Optional[BaseException] = None

        # Paths for IO from the ZMQ socket.
        input_path = get_open_zmq_ipc_path()
        output_path = get_open_zmq_ipc_path()

        # Start MQLLMEngine in a background process.
        self.engine_proc_handler = BackgroundProcHandle(
            input_path=input_path,
            output_path=output_path,
            process_name="MQLLMEngine",
            target_fn=MQLLMEngine.run_mq_llm_engine,
            process_kwargs={
                "engine_args": engine_args,
                "usage_context": usage_context,
            })

        # Get the tracing flag from the startup RPC.
        if (self.engine_proc_handler.data is None
                or "is_tracing_enabled" not in self.engine_proc_handler.data):
            raise ValueError(
                "Expected MQLLMEngine to send `is_tracing_enabled: bool` "
                f"to ready pipe, but got {self.engine_proc_handler.data}")
        self.tracing_flag = self.engine_proc_handler.data["is_tracing_enabled"]

        # Get the configs.
        engine_config = engine_args.create_engine_config()
        self.model_config = engine_config.model_config
        self.decoding_config = engine_config.decoding_config

        # Create the tokenizer group.
        self.tokenizer = init_tokenizer_from_configs(
            model_config=self.model_config,
            scheduler_config=engine_config.scheduler_config,
            parallel_config=engine_config.parallel_config,
            lora_config=engine_config.lora_config)
        self.input_preprocessor = InputPreprocessor(self.model_config,
                                                    self.tokenizer)

        # MQLLMEngine IO.
        self.ctx = zmq.asyncio.Context()
        self.input_socket = make_zmq_socket(self.ctx, input_path,
                                            zmq.constants.PUSH)
        self.output_socket = make_zmq_socket(self.ctx, output_path,
                                             zmq.constants.PULL)

        # Loop to handle output of the LLMEngine periodically.
        # Started after the MQLLMEngine is ready so that we can
        # build the Client in an executor to enable clean shutdown.
        self.output_queues: Dict[str, asyncio.Queue] = {}
        self.output_loop: Optional[asyncio.Task] = None

    @staticmethod
    def is_unsupported_config(engine_args: AsyncEngineArgs):
        # Pipeline parallel not yet supported
        return engine_args.pipeline_parallel_size > 1

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
                output = pickle.loads(message.buffer)

                # Occurs if there is an error in adding a new request.
                # Note: the server can keep running if this happens,
                # it only impacts a specific request.
                if isinstance(output, RPCError):

                    rpc_error: RPCError = output

                    # Put in the queue so it can be raised in generate().
                    queue = self.output_queues.get(rpc_error.request_id, None)
                    if queue is not None:
                        queue.put_nowait(rpc_error.exception)

                # One request output for each item in the batch.
                elif isinstance(output, List):
                    request_outputs: List[RequestOutput] = output

                    # Put each output into the appropriate steam.
                    for request_output in request_outputs:
                        queue = self.output_queues.get(
                            request_output.request_id, None)
                        if queue is not None:
                            queue.put_nowait(request_output)

                else:
                    self._set_errored(
                        ValueError(f"Unknown output in handler: {output}"))

        except asyncio.CancelledError:
            logger.debug("Shutting down MQLLMEngineClient output handler.")

    def shutdown(self):
        """Destroy the MQLLMEngine."""

        # Shutdown the background process.
        if hasattr(self, "engine_proc_handler"):
            self.engine_proc_handler.shutdown()

        # Close all sockets and terminate the context.
        if hasattr(self, "ctx"):
            self.ctx.destroy(linger=0)

        # Cancel background tasks.
        if hasattr(self, "output_loop") and self.output_loop:
            self.output_loop.cancel()

    def _set_errored(self, e: BaseException):
        logger.exception(repr(e))
        if self._errored_with is None:
            self._errored_with = e

    @staticmethod
    async def _send_one_way_rpc_request(request: RPC_REQUEST_T,
                                        socket: Socket):
        """Send one-way RPC request to trigger an action."""

        if socket.closed:
            raise MQClientClosedError()

        await socket.send_multipart((pickle.dumps(request), ))

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return self.input_preprocessor

    async def get_tokenizer(self, lora_request: Optional[LoRARequest] = None):
        return await self.tokenizer.get_lora_tokenizer_async(lora_request)

    async def get_decoding_config(self) -> DecodingConfig:
        return self.decoding_config

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def is_tracing_enabled(self) -> bool:
        return self.tracing_flag

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
    def engine_pid(self) -> int:
        return self.engine_proc_handler.pid

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

    @overload
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
        ...

    @overload
    @deprecated("'inputs' will be renamed to 'prompt")
    def generate(
        self,
        *,
        inputs: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        ...

    @deprecate_kwargs(
        "inputs",
        additional_message="Please use the 'prompt' parameter instead.",
    )
    def generate(
        self,
        prompt: Optional[PromptType] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        *,
        inputs: Optional[PromptType] = None  # DEPRECATED
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt to the LLM. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each input.
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

        # Start output handler loop on the first call.
        if self.output_loop is None:
            self.output_loop = asyncio.create_task(
                self.run_output_handler_loop())

        if inputs is not None:
            prompt = inputs
        assert (prompt is not None and sampling_params is not None
                and request_id is not None)

        return self._process_request(prompt, sampling_params, request_id,
                                     lora_request, trace_headers,
                                     prompt_adapter_request, priority)

    @overload
    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        ...

    @overload
    @deprecated("'inputs' will be renamed to 'prompt")
    def encode(
        self,
        *,
        inputs: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        ...

    @deprecate_kwargs(
        "inputs",
        additional_message="Please use the 'prompt' parameter instead.",
    )
    def encode(
        self,
        prompt: Optional[PromptType] = None,
        pooling_params: Optional[PoolingParams] = None,
        request_id: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        *,
        inputs: Optional[PromptType] = None  # DEPRECATED
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt to the LLM. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each input.
            pooling_params: The pooling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.

        Yields:
            The output `PoolingRequestOutput` objects from the LLMEngine
            for the request.
        """

        # Start output handler loop on the first call.
        if self.output_loop is None:
            self.output_loop = asyncio.create_task(
                self.run_output_handler_loop())

        if inputs is not None:
            prompt = inputs
        assert (prompt is not None and pooling_params is not None
                and request_id is not None)

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
                    default_guided_backend=(self.decoding_config.guided_decoding_backend
                        if self.decoding_config
                        else DecodingConfig.guided_decoding_backend),
                    model_config=self.model_config
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
