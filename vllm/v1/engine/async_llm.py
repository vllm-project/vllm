import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Mapping, Optional, Type, Union

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
from vllm.utils import get_open_zmq_ipc_path
from vllm.v1.engine.core_client import MultiprocessEngineCore
from vllm.v1.engine.detokenizer import DetokenizerClient
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
        assert start_engine_loop

        self.warned = False
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

        # RequestId -> OutputQueue.
        self.rid_to_queue: Dict[str, asyncio.Queue[RequestOutput]] = {}
        # List of cancelled request ids to be aborted.
        self.client_aborted_requests: List[str] = []

        # Processor (converts Inputs --> EngineRequest).
        self.processor = Processor(vllm_config.model_config,
                                   vllm_config.lora_config, self.tokenizer,
                                   input_registry)


        # IPC paths.
        engine_core_outputs_path = get_open_zmq_ipc_path()
        engine_core_inputs_path = get_open_zmq_ipc_path()

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput).
        self.detokenizer = DetokenizerClient(
            engine_core_outputs_path=engine_core_outputs_path,
            engine_core_inputs_path=engine_core_inputs_path,
            tokenizer_name=vllm_config.model_config.tokenizer,
            tokenizer_mode=vllm_config.model_config.tokenizer_mode,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
            revision=vllm_config.model_config.tokenizer_revision,
        )

        # EngineCore (starts the engine in background process).
        self.engine_core = MultiprocessEngineCore(
            input_path=engine_core_inputs_path,
            output_path=engine_core_outputs_path,
            vllm_config=vllm_config,
            executor_class=executor_class,
            usage_context=usage_context,
        )

        # self.output_handler: Optional[asyncio.Task] = None
        self.to_create_loop = True

    def __del__(self):
        self.shutdown()

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[VllmConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "AsyncLLMEngine":
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
        """Shutdown, cleaning up the background proc and IPC."""

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()
        
        if detokenizer := getattr(self, "detokenizer", None):
            detokenizer.shutdown()

        if handler := getattr(self, "output_handler", None):
            handler.cancel()

    @classmethod
    def _get_executor_cls(cls, vllm_config: VllmConfig) -> Type[Executor]:
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

        # 1) Convert Input --> EngineRequest.
        engine_request = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)
        
        # 2) Create Queue (output_handler pushes, generate pulls)
        self.rid_to_queue[request_id] = asyncio.Queue()

        # 3) Send to Detokenizer.
        await self.detokenizer.add_request_async(engine_request)

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
            * 1) Make a queue corresponding to the Request.
            # 2) Processing the Input.
            * 3) Adding the Request to the Detokenize + EngineCore.

        The output_handler() loop runs in a background task, pulling
        from Detokenizer and pushing to the per request queue.

        The generate() pulls from the per requests queue and yeilds
        to the caller which iterates the AsyncGenerator.
        """

        # We start the output_handler on the first call to generate() so that
        # we can call __init__ before the event loop starts, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        # if self.output_handler is None:
        if self.to_create_loop:
            import signal
            def signal_handler(self, signum=None, frame=None):
                logger.warning(
                    f"SIGTERM received. {signum=} {frame=}. Draining requests and shutting down..."
            )

            self.to_create_loop = False
            loop = asyncio.get_event_loop()
            loop.create_task(self._run_output_handler())
            loop.add_signal_handler(signal.SIGTERM, signal_handler)

        queue = await self.add_request(
            request_id,
            prompt,
            sampling_params,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )

        while True:
            try:
                out = await asyncio.wait_for(queue.get(), timeout=4)

                q_size = queue.qsize()
                # if q_size > 0:
                #     logger.info(f"{q_size=}")
                if out.finished:
                    del self.rid_to_queue[request_id]
                    yield out
                    break

                yield out

            except asyncio.TimeoutError:
                # TODO(rob): do request cancellation checking here.
                # logger.debug("Timeout waiting for %s", request_id)
                continue
                

    # async def _process_cancellations(self) -> None:
    #     """
    #     Process requests cancelled from user disconnecting.

    #     When a client disconnects, AsyncStream._cancel() is called.
    #     We passed a callback to AsyncStream(), which appends to 
    #     self.client_aborted_requests.

    #     As a result, if any requests are canceled from the user side
    #     the request_id will show up in self.client_aborted_requests.
    #     """

    #     # Avoid streams having circular ref to parent AsyncLLM object.
    #     if not self.client_aborted_requests:
    #         return
    #     reqs_to_abort = self.client_aborted_requests.copy()
    #     self.client_aborted_requests.clear()

    #     # Remove from Detokenizer.
    #     self.detokenizer.abort_requests(reqs_to_abort)

    #     # Remove from RequestStreams.
    #     for request_id in reqs_to_abort:
    #         if self.log_requests:
    #             logger.info("User-cancelled request %s.", request_id)
    #         self._finish_stream(request_id)

    #     # Remove from EngineCore.
    #     await self.engine_core.abort_requests_async(reqs_to_abort)

    async def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        epoch = 0
        while True:
            logger.info(f"EPOCH: {epoch}")
            epoch+=1

            # 1) Pull outputs from the Detokenizer.
            outputs: List[RequestOutput] = await self.detokenizer.output_socket.recv_pyobj()

            for out in outputs:
                if out.request_id not in self.rid_to_queue:
                    raise RuntimeError(f"{out.request_id} "
                                        "not in RequestStates")

                self.rid_to_queue[out.request_id].put_nowait(out)

            # 3) Abort any requests that finished due to stop strings.
            # await self.engine_core.abort_requests_async(reqs_to_abort)

            # 4) Abort any requests due to client cancellations.
            # TODO: send back to detokenizer if this fails.
            # await self._process_cancellations()

    async def abort(self, request_id: str) -> None:
        # Note: this is not used outside of testing.
        raise ValueError("Not Supported on V1 yet.")

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
        await self.engine_core.profile_async(True)

    async def stop_profile(self) -> None:
        await self.engine_core.profile_async(False)

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


# Retain V0 name for backwards compatibility.
AsyncLLMEngine = AsyncLLM
