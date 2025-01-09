import asyncio
import os
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
from vllm.utils import kill_process_tree
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.detokenizer import Detokenizer
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

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput).
        self.detokenizer = Detokenizer(
            tokenizer_name=vllm_config.model_config.tokenizer,
            tokenizer_mode=vllm_config.model_config.tokenizer_mode,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
            revision=vllm_config.model_config.tokenizer_revision,
        )

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=True,
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
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

        executor_class = Executor.get_class(vllm_config)

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

        if handler := getattr(self, "output_handler", None):
            handler.cancel()

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

        # 3) Add the request to Detokenizer (this process).
        self.detokenizer.add_request(request)

        # 4) Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

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
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() if q.qsize() > 0 else await q.get()

                # Note: both Detokenizer and EngineCore handle their
                # own request cleanup based on finished.
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
                # 1) Pull EngineCoreOutput from the EngineCore.
                outputs = await self.engine_core.get_output_async()

                # 2) Detokenize based on the output.
                request_outputs, reqs_to_abort = self.detokenizer.step(outputs)

                # 3) Put the RequestOutputs into the per-request queues.
                self._process_request_outputs(request_outputs)

                # 4) Abort any requests that finished due to stop strings.
                await self.engine_core.abort_requests_async(reqs_to_abort)

        except Exception as e:
            logger.exception("EngineCore output handler hit an error: %s", e)
            kill_process_tree(os.getpid())

    async def abort(self, request_id: str) -> None:
        """Abort RequestId in self, detokenizer, and engine core."""

        request_ids = [request_id]
        await self.engine_core.abort_requests_async(request_ids)
        self.detokenizer.abort_requests(request_ids)

        # If a request finishes while we await then the request_id
        # will be removed from the tracked queues before we get here.
        if request_id in self.rid_to_queue:
            del self.rid_to_queue[request_id]

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
        assert lora_request is None
        return self.detokenizer.tokenizer

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
