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
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_open_zmq_ipc_path
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.engine.detokenizer import DetokenizerClient
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)

WAITING_TIMEOUT_MS = 5


@dataclass
class RequestState:
    """
    RequestState manages concurrency between:
        * the output_handler(), which pulls outputs from EngineCore
        * the per-request generate(), which yields to the API server

    The output_handler adds new RequestOutputs to out_list and sets the 
    asyncio event, notifying the generate() that there is work to do.

    generate() waits on the asyncio event and yields the data from 
    out_list back to the caller generate()
    """

    event: asyncio.Event
    out_list: List[RequestOutput]

    @classmethod
    def new(cls) -> "RequestState":
        return cls(asyncio.Event(), [])


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

        # RequestId -> RequestState.
        self.rid_to_state: Dict[str, RequestState] = {}
        # List of cancelled request ids to be aborted.
        self.client_aborted_requests: List[str] = []

        # Processor (converts Inputs --> EngineCoreRequests).
        self.processor = Processor(vllm_config.model_config,
                                   vllm_config.lora_config, self.tokenizer,
                                   input_registry)


        # IPC path for EngineCore -> Detokenizer.
        engine_core_outputs_path = get_open_zmq_ipc_path()

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput).
        self.detokenizer = DetokenizerClient(
            engine_core_outputs_path=engine_core_outputs_path,
            tokenizer_name=vllm_config.model_config.tokenizer,
            tokenizer_mode=vllm_config.model_config.tokenizer_mode,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
            revision=vllm_config.model_config.tokenizer_revision,
        )

        # EngineCore (starts the engine in background process).
        self.engine_core = AsyncMPClient(
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
    ) -> RequestState:
        """Add new request to the AsyncLLM."""

        # 1) Add to RequestState tracker. The "event" is used to manage
        # concurrency between generate() and output_handler().
        self.rid_to_state[request_id] = RequestState.new()

        # 2) Convert input --> DetokenizerRequest / EngineCoreRequest.
        detokenizer_req, engine_core_req = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)

        # 3) Add the DetokenizerRequest to Detokenizer.
        # TODO: sending these separately is a race condition. We should instead
        # have the EngineCore do the "AddRequest" logic.
        await self.detokenizer.add_request_async(detokenizer_req)

        # 4) Add the EngineCoreRequest to EngineCore.
        await self.engine_core.add_request_async(engine_core_req)

        return self.rid_to_state[request_id]

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
            * 1) Make RequestState corresponding to the Request.
            # 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        The output_handler() loop runs in a background task, pulling from
        EngineCore and updating the RequestState and setting the asyncio event.

        The caller of generate() waits on the asyncio event and forwards
        the latest RequestOutput back to the caller. 
        """

        # DELTA streaming is not supported due to dynamic chunking.
        assert (sampling_params.output_kind == RequestOutputKind.CUMULATIVE
                or sampling_params.output_kind == RequestOutputKind.FINAL_ONLY)

        # We start the output_handler on the first call to generate() so that
        # we can call __init__ before the event loop starts, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        # if self.output_handler is None:
        if self.to_create_loop:
            loop = asyncio.get_event_loop()
            print(f"{loop=}")
            loop.create_task(self._run_output_handler())
            self.to_create_loop = False
            # self.output_handler = asyncio.create_task(
            #     self._run_output_handler())
            

        state = await self.add_request(
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
                await asyncio.wait_for(state.event.wait(),
                                       timeout=WAITING_TIMEOUT_MS)
                logger.info(f"{request_id} woke up.")

                # NOTE(rob): out_list can have more than one item. However,
                # in the streaming case, we use RequestOutputKind.CUMULATIVE,
                # which has the full generated text output (not just the text
                # corresponding to the last token). So, we can just send the
                # last RequestOutput and the API Client handles converting into
                # a delta text. This way we do "dynamic chunked streaming", such
                # that the API client does not fall behind the EngineCor,
                # which happens at high QPS otherwise.
                out = state.out_list[-1]
                if len(state.out_list) > 10:
                    logger.info(f"{len(state.out_list)=}")

            except asyncio.TimeoutError:
                # TODO(rob): do request cancellation checking here.
                logger.info("Timeout waiting for %s", request_id)
                continue

            state.out_list = []
            if out.finished:
                del self.rid_to_state[request_id]
                yield out
                break

            state.event.clear()
            yield out

    async def _process_cancellations(self) -> None:
        """
        Process requests cancelled from user disconnecting.

        When a client disconnects, AsyncStream._cancel() is called.
        We passed a callback to AsyncStream(), which appends to 
        self.client_aborted_requests.

        As a result, if any requests are canceled from the user side
        the request_id will show up in self.client_aborted_requests.
        """

        # Avoid streams having circular ref to parent AsyncLLM object.
        if not self.client_aborted_requests:
            return
        reqs_to_abort = self.client_aborted_requests.copy()
        self.client_aborted_requests.clear()

        # Remove from Detokenizer.
        self.detokenizer.abort_requests(reqs_to_abort)

        # Remove from RequestStreams.
        for request_id in reqs_to_abort:
            if self.log_requests:
                logger.info("User-cancelled request %s.", request_id)
            self._finish_stream(request_id)

        # Remove from EngineCore.
        await self.engine_core.abort_requests_async(reqs_to_abort)

    def _process_request_outputs(self, request_outputs: List[RequestOutput]):
        """Process outputs by putting them into per-request AsyncStreams."""

        for request_output in request_outputs:
            if request_output.request_id not in self.rid_to_state:
                raise RuntimeError(f"{request_output.request_id} "
                                    "not in RequestStates")

            # Update the RequestState and alert generate() that there
            # is a RequestOutput ready to return to the user.
            state = self.rid_to_state[request_output.request_id]
            state.out_list.append(request_output)
            state.event.set()

    async def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""
        # idx = 0

        # import pyinstrument

        # prof = pyinstrument.Profiler()
        # prof.start()
        # i = 0
        try:
            while True:
                # 1) Pull outputs from the Detokenizer.
                request_outputs, reqs_to_abort = (
                    await self.detokenizer.get_output_async())
                logger.info("AsyncLLM")
                # logger.info(f"RECV: {idx}")
                # idx+=1

                # 2) Put the RequestOutputs into the per-request AsyncStreams.
                self._process_request_outputs(request_outputs)

                # 3) Abort any requests that finished due to stop strings.
                # await self.engine_core.abort_requests_async(reqs_to_abort)

                # 4) Abort any requests due to client cancellations.
                # TODO: send back to detokenizer if this fails.
                # await self._process_cancellations()

        # except KeyboardInterrupt:
        #     prof.stop()
        #     prof.write_html("output_handler.prof")

        except Exception as e:
            logger.error(e)
            raise e

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
