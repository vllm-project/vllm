# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Mapping, Optional, Tuple, Type, Union

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
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParallelSamplingRequest
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
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # Bookkeeping for parallel sampling requests
        # - parent req ID -> parent request manager
        self.parallel_parent_reqs: Dict[str, ParallelSamplingRequest] = {}
        # - child req ID -> (child req index, parent req ID)
        self.parallel_child_reqs: Dict[str, Tuple[int, str]] = {}
        # - flag to reset parallel sampling bookkeeping logic
        #   between engine runs
        self._do_reset_parallel_sampling = False

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

    def get_num_unfinished_requests(self) -> int:
        num_core_reqs = self.output_processor.get_num_unfinished_requests()
        num_child_reqs = self._num_parallel_sampling_child_requests()
        num_parent_reqs = self._num_parallel_sampling_requests()
        return num_core_reqs + num_parent_reqs - num_child_reqs

    def has_unfinished_requests(self) -> bool:
        return self.output_processor.has_unfinished_requests()

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

    def abort_request(self, request_ids: List[str]) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        self.engine_core.abort_requests(request_ids)
        self.output_processor.abort_requests(request_ids)

    def _reset_parallel_sampling(self) -> None:
        """Reset parallel sampling logic"""
        self.parallel_parent_reqs.clear()
        self.parallel_child_reqs.clear()
        self._do_reset_parallel_sampling = False

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
        if self._do_reset_parallel_sampling:
            # Reset parallel sampling logic between
            # LLM.generate() calls
            self._reset_parallel_sampling()
        # Handle parallel sampling requests differently.
        _add_request = (self._add_request if params is None
                        or isinstance(params, PoolingParams) or params.n == 1
                        else self._add_request_parallel_sampling)
        return _add_request(request_id=request_id,
                            prompt=prompt,
                            params=params,
                            arrival_time=arrival_time,
                            lora_request=lora_request,
                            trace_headers=trace_headers,
                            prompt_adapter_request=prompt_adapter_request,
                            priority=priority)

    def _add_request_parallel_sampling(
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
        """Add request, `n>1`"""
        req_mgr = ParallelSamplingRequest(request_id, params)
        self.parallel_parent_reqs[request_id] = req_mgr
        # Add n child requests with unique request IDs & random seeds and n=1
        for idx in range(req_mgr.n):
            c_req_id, c_params = req_mgr.get_child_info(idx)
            self.parallel_child_reqs[c_req_id] = (idx, request_id)
            self._add_request(request_id=c_req_id,
                              prompt=prompt,
                              params=c_params,
                              arrival_time=arrival_time,
                              lora_request=lora_request,
                              trace_headers=trace_headers,
                              prompt_adapter_request=prompt_adapter_request,
                              priority=priority)

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

    def _aggregate_parallel_sampling_outputs(
        self,
        outputs: List[RequestOutput],
    ) -> List[RequestOutput]:
        """Build parallel sampling request outputs.
        
        Extract child request outputs, aggregate them
        into parent request output, and return parent
        output when complete.

        Do not modify `n=1` requests.

        Args:
          outputs: step request outputs. Mix of child request
                   outputs & `n=1` request outputs.

        Return:
          List of parallel sampling parent request outputs &
          unmodified `n=1` request outputs passed-thru from input.
        """
        agg_outputs = []
        for c_out in outputs:
            c_req_id = c_out.request_id
            if cdx_req_id := self.parallel_child_reqs.get(c_req_id, None):
                # For each parallel sampling child request output:
                (cdx, req_id) = cdx_req_id
                req_mgr = self.parallel_parent_reqs[req_id]
                # Update parallel sampling request
                if out := req_mgr._process_output(c_out, cdx):
                    # Return parent request output if complete;
                    # cleanup parent request bookkeeping.
                    agg_outputs.append(out)
                    del self.parallel_parent_reqs[req_id]
                # Cleanup child request bookkeeping.
                del self.parallel_child_reqs[c_req_id]
            else:
                # Not a parallel sampling request output
                agg_outputs.append(c_out)
        return agg_outputs

    def _num_parallel_sampling_requests(self) -> int:
        return len(self.parallel_parent_reqs)

    def _num_parallel_sampling_child_requests(self) -> int:
        return len(self.parallel_child_reqs)

    def step(self) -> List[RequestOutput]:
        num_parallel_reqs = self._num_parallel_sampling_requests()

        # Ensure that parallel sampling logic gets reset after the
        # engine finishes processing this batch
        if self.parallel_parent_reqs:
            self._do_reset_parallel_sampling = True

        # 1) Get EngineCoreOutput from the EngineCore.
        outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        processed_outputs = self.output_processor.process_outputs(
            outputs.outputs)

        # 3) Abort any reqs that finished due to stop strings.
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        request_outputs = processed_outputs.request_outputs
        if num_parallel_reqs > 0 and len(request_outputs) > 0:
            # Process parallel sampling child request outputs
            return self._aggregate_parallel_sampling_outputs(request_outputs)
        else:
            return request_outputs

    def get_model_config(self):
        return self.model_config

    def start_profile(self):
        self.engine_core.profile(True)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_prefix_cache(self):
        self.engine_core.reset_prefix_cache()

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
