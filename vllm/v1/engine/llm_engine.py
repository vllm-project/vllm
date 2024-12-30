import weakref
from typing import Dict, List, Mapping, Optional, Type, Union

import zmq
from typing_extensions import TypeVar

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.envs import VLLM_ENABLE_V1_MULTIPROCESSING
from vllm.inputs import INPUT_REGISTRY, InputRegistry, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import (
    BaseTokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.engine.core import (EngineCore, EngineCoreClient, EngineCoreProc,
                                 InprocEngineCoreClient, MpEngineCoreClient)
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)

_G = TypeVar("_G", bound=BaseTokenizerGroup, default=BaseTokenizerGroup)


class LLMEngine:

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)

        self.model_config = vllm_config.model_config

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)
        self.tokenizer.ping()

        # Processor (convert Inputs --> EngineCoreRequests)
        self.processor = Processor(model_config=vllm_config.model_config,
                                   cache_config=vllm_config.cache_config,
                                   lora_config=vllm_config.lora_config,
                                   tokenizer=self.tokenizer,
                                   input_registry=input_registry,
                                   mm_registry=mm_registry)

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput).
        # Note(rob): does it ever make sense to run this in background?
        self.detokenizer = Detokenizer(
            tokenizer_name=vllm_config.model_config.tokenizer,
            tokenizer_mode=vllm_config.model_config.tokenizer_mode,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
            revision=vllm_config.model_config.tokenizer_revision,
        )

        self.engine_core_client: EngineCoreClient

        # EngineCore (converts EngineCoreRequests --> EngineCoreOutputs)
        if multiprocess_mode:
            # Setup ZMQ for IPC.
            input_path = get_open_zmq_ipc_path()
            output_path = get_open_zmq_ipc_path()
            self.ctx = zmq.Context(io_threads=2)  # type: ignore[attr-defined]
            self.to_engine_core = make_zmq_socket(self.ctx, input_path,
                                                  zmq.constants.PUSH)
            self.from_engine_core = make_zmq_socket(self.ctx, output_path,
                                                    zmq.constants.PULL)

            # Make EngineCore in background process + make client.
            handle = self.engine_core_handle = EngineCoreProc.make_process(
                vllm_config=vllm_config,
                executor_class=executor_class,
                input_path=input_path,
                output_path=output_path,
                log_stats=log_stats,
            )
            self.engine_core_client = MpEngineCoreClient(
                input_socket=self.to_engine_core,
                output_socket=self.from_engine_core,
                proc_handle=handle,
            )

        else:
            # Make EngineCore in process + make client.
            self.engine_core = EngineCore(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
            )
            self.engine_core_client = InprocEngineCoreClient(
                engine_core=self.engine_core)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = cls._get_executor_cls(vllm_config)

        if VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(vllm_config=vllm_config,
                   executor_class=executor_class,
                   log_stats=not engine_args.disable_log_stats,
                   usage_context=usage_context,
                   stat_loggers=stat_loggers,
                   multiprocess_mode=enable_multiprocessing)

    @classmethod
    def _get_executor_cls(cls, vllm_config: VllmConfig) -> Type[Executor]:
        executor_class: Type[Executor]
        distributed_executor_backend = (
            vllm_config.parallel_config.distributed_executor_backend)
        if distributed_executor_backend == "ray":
            from vllm.v1.executor.ray_executor import RayExecutor
            executor_class = RayExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            executor_class = MultiprocExecutor
        else:
            assert (distributed_executor_backend is None)
            from vllm.v1.executor.uniproc_executor import UniprocExecutor
            executor_class = UniprocExecutor

        return executor_class

    def get_num_unfinished_requests(self) -> int:
        return self.detokenizer.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        return self.detokenizer.has_unfinished_requests()

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

    def abort_request(self, request_ids: List[str]) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        self.engine_core.abort_requests(request_ids)
        self.detokenizer.abort_requests(request_ids)

    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:

        # 1) Process raw inputs into the request.
        request = self.processor.process_inputs(request_id, prompt, params,
                                                arrival_time, lora_request,
                                                trace_headers,
                                                prompt_adapter_request,
                                                priority)

        # 2) Add the request to Detokenizer.
        self.detokenizer.add_request(request)

        # 3) Add the request to EngineCore.
        self.engine_core.add_request(request)

    def step(self) -> List[RequestOutput]:

        # 1) Get EngineCoreOutput from the EngineCore.
        engine_core_outputs = self.engine_core_client.get_output()

        # 2) Detokenizer the EngineCoreOutput.
        request_outputs, requests_to_abort = self.detokenizer.step(
            engine_core_outputs)

        # 3) Abort requests that finished due to stopping criteria.
        if requests_to_abort:
            self.abort_request(requests_to_abort)

        return request_outputs

    def get_model_config(self):
        return self.model_config

    def start_profile(self):
        self.engine_core_client.profile(True)

    def stop_profile(self):
        self.engine_core_client.profile(False)

    def get_tokenizer_group(
        self,
        group_type: Type[_G] = BaseTokenizerGroup,
    ) -> _G:
        tokenizer_group = self.tokenizer

        if tokenizer_group is None:
            raise ValueError("Unable to get tokenizer because "
                             "skip_tokenizer_init is True")
        if not isinstance(tokenizer_group, group_type):
            raise TypeError("Invalid type of tokenizer group. "
                            f"Expected type: {group_type}, but "
                            f"found type: {type(tokenizer_group)}")

        return tokenizer_group

    def shutdown(self):
        """Shutdown EngineCore."""

        if engine_core_client := getattr(self, "engine_core_client", None):
            engine_core_client.shutdown()

        if engine_core_handle := getattr(self, "engine_core_handle", None):
            engine_core_handle.shutdown()

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()
