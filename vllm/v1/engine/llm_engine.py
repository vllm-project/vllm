import pickle
from typing import Any, Dict, List, Mapping, Optional, Set, Type, Union

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
from vllm.utils import get_open_zmq_ipc_path
from vllm.v1.engine import EngineRequestType
from vllm.v1.engine.core import EngineCore, MPEngineCoreClient
from vllm.v1.engine.detokenizer import Detokenizer, MPDetokenizerClient
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.ray_utils import initialize_ray_cluster
from vllm.v1.utils import make_zmq_socket

logger = init_logger(__name__)

_G = TypeVar("_G", bound=BaseTokenizerGroup, default=BaseTokenizerGroup)


class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

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

        self.multiprocess_mode = multiprocess_mode
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

        if self.multiprocess_mode:
            # Keep track of active requests.
            self.running_requests: Set[str] = set()

            # IPC paths.
            to_detokenizer_path = get_open_zmq_ipc_path()
            to_engine_core_path = get_open_zmq_ipc_path()
            to_llm_engine_path = get_open_zmq_ipc_path()

            # Detokenizer IPC.
            self.ctx = zmq.Context(io_threads=2)  # type: ignore[attr-defined]
            self.from_detokenizer = make_zmq_socket(self.ctx,
                                                    to_llm_engine_path,
                                                    zmq.constants.PULL)
            self.to_detokenizer = make_zmq_socket(self.ctx,
                                                  to_detokenizer_path,
                                                  zmq.constants.PUSH)

            # Detokenizer (background process).
            self.detokenizer_client = MPDetokenizerClient(
                output_path=to_llm_engine_path,
                input_path=to_detokenizer_path,
                to_engine_core_path=to_engine_core_path,
                tokenizer_name=vllm_config.model_config.tokenizer,
                tokenizer_mode=vllm_config.model_config.tokenizer_mode,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
                revision=vllm_config.model_config.tokenizer_revision,
            )

            # EngineCore (background process).
            self.engine_core_client = MPEngineCoreClient(
                input_path=to_engine_core_path,
                output_path=to_detokenizer_path,
                vllm_config=vllm_config,
                executor_class=executor_class,
                usage_context=usage_context,
            )

        else:
            # Detokenizer (in process).
            self.detokenizer = Detokenizer(
                tokenizer_name=vllm_config.model_config.tokenizer,
                tokenizer_mode=vllm_config.model_config.tokenizer_mode,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
                revision=vllm_config.model_config.tokenizer_revision,
            )

            # EngineCore (in process).
            self.engine_core = EngineCore(
                vllm_config=vllm_config,
                executor_class=executor_class,
                usage_context=usage_context,
            )

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
            initialize_ray_cluster(vllm_config.parallel_config)
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
        if self.multiprocess_mode:
            return len(self.running_requests)
        else:
            return self.detokenizer.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

    def abort_request(self, request_ids: List[str]) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        assert not self.multiprocess_mode
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

        logger.info("Added request.")

        # Process raw inputs into the request.
        engine_request = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)

        # Add processed input to system.
        if self.multiprocess_mode:
            assert engine_request.request_id not in self.running_requests
            self.running_requests.add(engine_request.request_id)
            # Send to Detokenizer (which forwards to EngineCore).
            # Note: we forward the message rather than sending
            # to each process separately to avoid race conditions.
            self._send_to_detokenizer(engine_request)
        else:
            # Add directly to Detokenizer and EngineCore.
            self.detokenizer.add_request(engine_request)
            self.engine_core.add_request(engine_request)

    def step(self) -> List[RequestOutput]:

        logger.info("Called step.")
        if self.multiprocess_mode:
            # Get next output from the Detokenizer.
            request_outputs: List[
                RequestOutput] = self.from_detokenizer.recv_pyobj()

            # Removed finished requests from the state tracker.
            for out in request_outputs:
                if out.finished:
                    self.running_requests.remove(out.request_id)

        else:
            # Step EngineCore and Detokenizer.
            engine_core_outputs = self.engine_core.step()
            request_outputs, requests_to_abort = self.detokenizer.step(
                engine_core_outputs)

            # Abort any requests that hit a stop string.
            if requests_to_abort:
                self.abort_request(requests_to_abort)

        return request_outputs

    def _send_to_detokenizer(self, object: Any):
        """Send object to Detokenizer with a FROM_ENGINE flag."""

        msg = (EngineRequestType.FROM_ENGINE.value, pickle.dumps(object))
        self.to_detokenizer.send_multipart(msg, copy=False)

    def get_model_config(self):
        return self.model_config

    def start_profile(self):
        self.engine_core.profile(True)

    def stop_profile(self):
        self.engine_core.profile(False)

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

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if engine_core_client := getattr(self, "engine_core_client", None):
            engine_core_client.shutdown()

        if detokenizer_client := getattr(self, "detokenizer_client", None):
            detokenizer_client.shutdown()
