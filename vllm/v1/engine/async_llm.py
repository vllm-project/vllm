import asyncio
import os
import pickle
import signal
import weakref
from typing import (Any, AsyncGenerator, Dict, List, Mapping, Optional, Type,
                    Union)

import zmq
import zmq.asyncio

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.protocol import EngineClient
from vllm.inputs import INPUT_REGISTRY, InputRegistry, PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (get_open_zmq_ipc_path, kill_process_tree,
                        make_zmq_socket)
from vllm.v1.engine import EngineCoreAbort, EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.detokenizer import DetokenizerProc
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)


class AsyncLLM(EngineClient):

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
    ) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)

        # The child processes will send SIGQUIT when unrecoverable
        # errors happen. We kill the process tree here so that the
        # stack trace is very evident.
        # TODO: rather than killing the main process, we should
        # figure out how to raise an AsyncEngineDeadError and
        # handle at the API server level so we can return a better
        # error code to the clients calling VLLM.
        def sigquit_handler(signum, frame):
            logger.fatal(
                "AsyncLLM got SIGQUIT from worker processes, shutting "
                "down. See stack trace above for root cause issue.")
            kill_process_tree(os.getpid())

        signal.signal(signal.SIGQUIT, sigquit_handler)

        assert start_engine_loop

        self.log_requests = log_requests
        self.log_stats = log_stats
        self.stat_loggers = stat_loggers
        self.model_config = vllm_config.model_config

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)
        self.tokenizer.ping()

        # Request streams (map of request_id -> queue).
        self.rid_to_queue: Dict[str, asyncio.Queue] = {}

        # Processor (converts Inputs --> EngineCoreRequests).
        self.processor = Processor(
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            lora_config=vllm_config.lora_config,
            tokenizer=self.tokenizer,
            input_registry=input_registry,
        )

        # Setup ZMQ IPC. Message flow is:
        # AsyncLLM <-> Detokenizer <-> EngineCore
        to_engine_core_path = get_open_zmq_ipc_path()
        to_detokenizer_path = get_open_zmq_ipc_path()
        from_detokenizer_path = get_open_zmq_ipc_path()
        self.ctx = zmq.asyncio.Context(io_threads=2)
        self.to_detokenizer = make_zmq_socket(self.ctx, to_detokenizer_path,
                                              zmq.constants.PUSH)
        self.from_detokenizer = make_zmq_socket(self.ctx,
                                                from_detokenizer_path,
                                                zmq.constants.PULL)

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput).
        self.detokenizer_handle = DetokenizerProc.make_process(
            input_path=to_detokenizer_path,
            output_path=from_detokenizer_path,
            to_engine_core_path=to_engine_core_path,
            tokenizer_name=self.model_config.tokenizer,
            tokenizer_mode=self.model_config.tokenizer_mode,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.revision,
        )

        # EngineCore (starts the engine in background process).
        # (Gets input from Detokenizer, sends outputs to Detokenizer).
        self.engine_core_handle = EngineCoreProc.make_process(
            vllm_config=vllm_config,
            executor_class=executor_class,
            input_path=from_detokenizer_path,
            output_path=to_detokenizer_path,
            log_stats=log_stats,
        )

        self.output_handler: Optional[asyncio.Task] = None

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[VllmConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        if engine_config is None:
            vllm_config = engine_args.create_engine_config(usage_context)
        else:
            vllm_config = engine_config

        executor_class = cls._get_executor_cls(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def shutdown(self):
        """Shutdown, cleaning up the background procs and IPC."""
        # ZMQ.
        self.ctx.destroy(linger=0)

        # EngineCore background process.
        if hasattr(self, "engine_core_handle"):
            self.engine_core_handle.shutdown()

        # Detokenizer background process.
        if hasattr(self, "engine_core_handle"):
            self.engine_core_handle.shutdown()

        # Output handler background task.
        if hasattr(self, "output_handler") and self.output_handler:
            self.output_handler.cancel()

    @staticmethod
    def _get_executor_cls(vllm_config: VllmConfig) -> Type[Executor]:
        executor_class: Type[Executor]
        distributed_executor_backend = (
            vllm_config.parallel_config.distributed_executor_backend)
        if distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            executor_class = MultiprocExecutor
        else:
            assert (distributed_executor_backend is None)
            from vllm.v1.executor.uniproc_executor import UniprocExecutor
            executor_class = UniprocExecutor
        return executor_class

    async def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> asyncio.Queue[RequestOutput]:
        """Add new request to the AsyncLLM."""

        # 1) Create a new output queue for the request.
        if request_id in self.rid_to_queue:
            raise ValueError(f"Request id {request_id} already running.")
        self.rid_to_queue[request_id] = asyncio.Queue()

        # 2) Convert Input --> Request.
        request = self.processor.process_inputs(request_id, prompt, params,
                                                arrival_time, lora_request,
                                                trace_headers,
                                                prompt_adapter_request,
                                                priority)

        # 3) Send to Detokenizer (which forwards to EngineCore).
        # note(rob): we forward the request rather than sending to each
        # process separately to avoid race conditions.
        await self._send_pyobj(self.to_detokenizer, request)

        if self.log_requests:
            logger.info("Added request %s.", request_id)

        return self.rid_to_queue[request_id]

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task, 
        pulling outputs from EngineCore and putting them into the 
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        try:
            # We start the output_handler on the first call to generate() so
            # we can call __init__ before the event loop, which enables us
            # to handle startup failure gracefully in the OpenAI server.
            if self.output_handler is None:
                self.output_handler = asyncio.create_task(
                    self._run_output_handler())

            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            while True:
                # note(rob): drain queue without await if possible
                # (avoids task switching under load for performance).
                out = q.get_nowait() if q.qsize() > 0 else await q.get()

                # notte(rob): both Detokenizer and EngineCore handle
                # their own request cleanup based on finished.
                if out.finished:
                    del self.rid_to_queue[request_id]
                    yield out
                    break

                yield out

        # If the request is disconnected by the client, the
        # generate() task will be canceled. So, we abort the
        # request if we end up here.
        except asyncio.CancelledError:
            await self.abort(request_id)
            raise

    def _process_request_outputs(self, request_outputs: List[RequestOutput]):
        """Process outputs by putting them into per-request queues."""

        for request_output in request_outputs:
            request_id = request_output.request_id

            # Note: it is possible a request was aborted and removed from
            # the state due to client cancellations, so if we encounter a
            # request id not in the state, we skip.
            if request_id in self.rid_to_queue:
                self.rid_to_queue[request_id].put_nowait(request_output)

    async def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        try:
            while True:
                # note(rob): use socket directly to avoid calling await multiple
                # times, which causes too much task switching at high QPS.
                outputs: List[RequestOutput] = []
                outputs = await self.from_detokenizer.recv_pyobj()

                for out in outputs:
                    # Note(rob): it is possible that a request was aborted
                    # due to cancellation, so we just skip if not found.
                    if out.request_id in self.rid_to_queue:
                        self.rid_to_queue[out.request_id].put_nowait(out)

        except Exception as e:
            logger.exception("EngineCore output handler hit an error: %s", e)
            kill_process_tree(os.getpid())

    async def abort(self, request_id: str) -> None:
        """Abort RequestId in self, detokenizer, and engine core."""

        # Alert detokenizer that we have an abort (message is forwarded
        # to the EngineCore).
        await self._send_pyobj(self.to_detokenizer,
                               EngineCoreAbort([request_id]))

        # If a request finishes while we await then the request_id
        # will be removed from the tracked queues before we get here.
        if request_id in self.rid_to_queue:
            del self.rid_to_queue[request_id]

    @staticmethod
    async def _send_pyobj(socket: zmq.asyncio.Socket, obj: Any):
        """Send object to Detokenizer with a FROM_ENGINE flag."""

        msg = (EngineCoreRequestType.FROM_ENGINE.value, pickle.dumps(obj))
        await socket.send_multipart(msg, copy=False)

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ):
        raise ValueError("Not Supported on V1 yet.")

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def get_decoding_config(self):
        raise ValueError("Not Supported on V1 yet.")

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return self.processor.input_preprocessor

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return self.tokenizer.get_lora_tokenizer(lora_request)

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(
        self,
        scheduler_outputs=None,
        model_output=None,
    ) -> None:
        logger.debug("Called do_log_stats.")

    async def check_health(self) -> None:
        logger.debug("Called check_health.")

    async def start_profile(self) -> None:
        pass

    async def stop_profile(self) -> None:
        pass

    @property
    def is_running(self) -> bool:
        return True

    @property
    def is_stopped(self) -> bool:
        return False

    @property
    def errored(self) -> bool:
        return False

    @property
    def dead_error(self) -> BaseException:
        return Exception()  # TODO: implement
