# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Mapping, Optional, Type, Union

from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase
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
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import SyncParallelSamplingManager
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor

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
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # Bookkeeping for parallel sampling requests
        self.parallel_manager = SyncParallelSamplingManager()

        # important: init dp group before init the engine_core
        self.parallel_config = vllm_config.parallel_config
        self.dp_enabled = self.parallel_config.data_parallel_size > 1  # noqa
        self.should_execute_dummy_batch = False
        if self.dp_enabled:
            self.dp_group = self.parallel_config.stateless_init_dp_group()

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

        # OutputProcessor (convert EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(self.tokenizer,
                                                log_stats=False)

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=False,  # FIXME: implement
        )

        if not multiprocess_mode:
            # for v0 compatibility
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

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
        executor_class = Executor.get_class(vllm_config)

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(vllm_config=vllm_config,
                   executor_class=executor_class,
                   log_stats=not engine_args.disable_log_stats,
                   usage_context=usage_context,
                   stat_loggers=stat_loggers,
                   multiprocess_mode=enable_multiprocessing)

    def get_num_unfinished_requests(self) -> int:
        return self.parallel_manager.get_num_unfinished_requests(
            self.output_processor.get_num_unfinished_requests())

    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        if not self.dp_enabled:
            return has_unfinished
        return self.has_unfinished_requests_dp(has_unfinished)

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished)
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

    def abort_request(self, request_ids: List[str]) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        self.engine_core.abort_requests(request_ids)
        self.output_processor.abort_requests(request_ids)

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
        """Add request."""
        kwargs = dict(request_id=request_id,
                      prompt=prompt,
                      params=params,
                      arrival_time=arrival_time,
                      lora_request=lora_request,
                      trace_headers=trace_headers,
                      prompt_adapter_request=prompt_adapter_request,
                      priority=priority)
        # Handle parallel sampling requests differently.
        if params is None or isinstance(params,
                                        PoolingParams) or params.n == 1:
            self._add_request(**kwargs)
        else:
            # Special handling for parallel sampling requests
            self.parallel_manager.add_request_parallel_sampling(
                add_request=self._add_request, **kwargs)

    def _add_request(
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
        """Add request, `n=1`"""
        # 1) Process raw inputs into the request.
        request = self.processor.process_inputs(request_id, prompt, params,
                                                arrival_time, lora_request,
                                                trace_headers,
                                                prompt_adapter_request,
                                                priority)

        # 2) Make a new RequestState and queue.
        self.output_processor.add_request(request)

        # 3) Add the request to EngineCore.
        self.engine_core.add_request(request)

    def step(self) -> List[RequestOutput]:

        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        processed_outputs = self.output_processor.process_outputs(
            outputs.outputs)

        # 3) Abort any reqs that finished due to stop strings.
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        request_outputs = processed_outputs.request_outputs

        # 4) Process unfinished parallel sampling requests
        return self.parallel_manager.step(request_outputs)

    def get_model_config(self):
        return self.model_config

    def start_profile(self):
        self.engine_core.profile(True)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_prefix_cache(self):
        self.engine_core.reset_prefix_cache()

    def sleep(self, level: int = 1):
        self.engine_core.sleep(level)

    def wake_up(self):
        self.engine_core.wake_up()

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
