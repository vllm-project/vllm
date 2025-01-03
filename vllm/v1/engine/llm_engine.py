import copy
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Type, Union

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
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)

_G = TypeVar("_G", bound=BaseTokenizerGroup, default=BaseTokenizerGroup)


def _none_safe_min(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return min(x, y)


def _none_safe_max(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return max(x, y)


def _none_safe_sum(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return x + y


@dataclass
class ParallelSampleChildRequestInfo:
    """Info for aggregating parallel sampling child requests under parent"""
    parent_req_id: str
    index: int


@dataclass
class ParallelSampleParentRequestInfo:
    """Parallel sampling parent request info"""
    n: int
    last_request_output: Optional[RequestOutput] = None
    n_aborted: int = 0
    n_finished: int = 0

    def gen_num_active_children(self):
        assert self.n >= self.n_finished + self.n_aborted
        return self.n - self.n_finished - self.n_aborted

    def incr_num_aborted(self):
        self.n_aborted += 1

    def get_num_active_children_incr_n_finished_if_true(
        self,
        child_req_finished: bool,
    ) -> int:
        if child_req_finished:
            self.n_finished += 1
        return self.gen_num_active_children()


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

        # TODO: Can we avoid this?
        self.model_config = vllm_config.model_config

        # Parallel sampling metadata
        # - Metadata for aggregating the child requests associated with a
        #   parent request
        self.child_req_id_to_parent_req_info: Dict[
            str, ParallelSampleChildRequestInfo] = {}
        # - Parent request metadata i.e. degree of parallelism and other
        #   characteristics
        self.parent_req_id_info: Dict[str,
                                      ParallelSampleParentRequestInfo] = {}

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

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput)
        self.detokenizer = Detokenizer(
            tokenizer_name=vllm_config.model_config.tokenizer,
            tokenizer_mode=vllm_config.model_config.tokenizer_mode,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
            revision=vllm_config.model_config.tokenizer_revision,
        )

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=False,
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

    def _get_num_core_unfinished_requests(self) -> int:
        """Total number of unfinished requests in engine core
        
        Does not account for parallel sampling, i.e. a request
        with `n=3` contributes `(3-n_complete)` to the total
        (the parent request
        does not count); an unfinished request with `n=1`
        contributes 1 to the total.

        Returns:
          Total requests in engine core
        """
        return self.detokenizer.get_num_unfinished_requests()

    def _get_num_parallel_sampling_parent_unfinished_requests(self) -> int:
        """Total number of requests with parallel sampling
        
        i.e. an unfinished request with `n=<blah>` counts as 1,
        all other requests count a 0.

        Returns:
          Number of parallel sampling parent requests
        """
        return len(self.parent_req_id_info)

    def _get_num_parallel_sampling_child_unfinished_requests(self) -> int:
        """Total number of parallel sampling child requests.
        
        i.e. an unfinished request with `n>1` counts as `(n-n_complete)`,
        all other requests count as 0.

        Returns:
          Number of parallel sampling child requests
        """
        return sum([
            preq_info.gen_num_active_children()
            for (_, preq_info) in self.parent_req_id_info.items()
        ])

    def get_num_unfinished_requests(self) -> int:
        """Number of unfinished requests.
        
        Each request submitted by the user counts as 1; the child requests
        spawned by parallel sampling requests are not reflected in this count.
        """
        return (self._get_num_core_unfinished_requests() -
                self._get_num_parallel_sampling_child_unfinished_requests() +
                self._get_num_parallel_sampling_parent_unfinished_requests())

    def has_unfinished_requests(self) -> bool:
        return self.detokenizer.has_unfinished_requests()

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

    def _forget_parallel_sample_parent_request(
        self,
        parent_req_id: str,
    ) -> None:
        self.parent_req_id_info.pop(parent_req_id, None)

    def _forget_parallel_sample_child_request(
        self,
        child_request_id: str,
    ) -> str:
        """Forget child request parallel sampling metadata, & its' parent's
        metadata if necessary.
        
        Parent request parallel sampling metadata is forgotten once all
        child requests are inactive.

        Args:
          child_request_id: id of finished child request
        """
        # Forget child request metadata
        parent_req_id = self.child_req_id_to_parent_req_info[
            child_request_id].parent_req_id
        self.child_req_id_to_parent_req_info.pop(child_request_id, None)
        return parent_req_id

    def _maybe_forget_parallel_sample_child_requests(
            self, possible_child_request_ids: List[str]) -> None:
        """When a request aborts, if it is a child of a parallel sampling
        request, forget its parallel sampling metadata. Apply this to a
        list of possible child request ids. If the request is not
        associated with parallel sampling, this method has no effect on
        it.
        
        Args:
          request_ids: list of possible child req ids
        """
        for possible_child_req_id in possible_child_request_ids:
            # Check if request is a parallel sampling child request
            if possible_child_req_id in self.child_req_id_to_parent_req_info:
                # If so, forget child request parallel sampling metadata
                parent_req_id = self._forget_parallel_sample_child_request(
                    possible_child_req_id)

                # Track parent request's remaining child requests & erase parent
                # request metadata if there are no remaining child requests
                self.parent_req_id_info[parent_req_id].incr_num_aborted()
                if self.parent_req_id_info[
                        parent_req_id].gen_num_active_children() < 1:
                    self._forget_parallel_sample_parent_request(parent_req_id)

    def abort_request(self, request_ids: List[str]) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        self.engine_core.abort_requests(request_ids)
        self.detokenizer.abort_requests(request_ids)
        self._maybe_forget_parallel_sample_child_requests(request_ids)

    def _register_parallel_sampling_parent_request(
        self,
        parent_req_id: str,
        parallel_sample_parent_req_info: ParallelSampleParentRequestInfo,
    ) -> None:
        """Register parallel sampling request (i.e. the parent request)"""
        self.parent_req_id_info[
            parent_req_id] = parallel_sample_parent_req_info

    def _register_parallel_sampling_child_request(
        self,
        parallel_sample_child_req_info: ParallelSampleChildRequestInfo,
    ) -> str:
        """Register association of parallel sampling child req with parent req.
        
        Generates a child request id

        Side effect: internal mapping from child req id -> parent req info
                     structure

        Returns:
          Child request id
        """
        parent_req_id = parallel_sample_child_req_info.parent_req_id
        index = parallel_sample_child_req_info.index
        child_req_id = f"{parent_req_id}_parallel_sample_{index}"
        self.child_req_id_to_parent_req_info[
            child_req_id] = parallel_sample_child_req_info
        return child_req_id

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
        if isinstance(params, SamplingParams) and params.n > 1:
            child_params = copy.copy(params)
            # Register parallel sampling request
            n = params.n
            self._register_parallel_sampling_parent_request(
                request_id, ParallelSampleParentRequestInfo(n))
            child_params.n = 1  # Engine core cannot see `n`
            for ndx in range(n):
                # Register child request with parent
                child_req_id = self._register_parallel_sampling_child_request(
                    ParallelSampleChildRequestInfo(request_id, ndx))
                # Recurse to add child request; `n=1` prevents further recursion
                self.add_request(
                    request_id=child_req_id,
                    prompt=prompt,
                    params=child_params,
                    arrival_time=arrival_time,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=priority,
                )
            # The top-level add_request call is done
            return

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

    def _is_parallel_sampling_child_request(
        self,
        possible_child_request_id: str,
    ) -> bool:
        return possible_child_request_id in self.child_req_id_to_parent_req_info

    def _maybe_get_parallel_sampling_child_request_info(
        self,
        possible_child_request_id: str,
    ) -> Optional[ParallelSampleChildRequestInfo]:
        return self.child_req_id_to_parent_req_info.get(
            possible_child_request_id, None)

    def _get_parallel_sampling_parent_request_info(
        self,
        parent_request_id: str,
    ) -> ParallelSampleParentRequestInfo:
        assert parent_request_id in self.parent_req_id_info, (
            "Not a parallel sampling request")
        return self.parent_req_id_info[parent_request_id]

    def _merge_parallel_sampling_child_request_output_in_place(
        self,
        parent_req_output: RequestOutput,
        child_req_output: RequestOutput,
        parent_req_info: ParallelSampleParentRequestInfo,
    ) -> None:
        # Parent is finished when all children are finished
        parent_req_output.finished = (
            parent_req_info.get_num_active_children_incr_n_finished_if_true(
                child_req_output.finished) < 1)
        p_met = parent_req_output.metrics
        c_met = child_req_output.metrics
        if p_met is None:
            # If current parent request metrics are `None`, update with this
            # child's metrics (which may also be None)
            parent_req_output.metrics = c_met
        elif c_met is not None:
            # Only merge in child request output metrics if the child request
            # output metrics are not `None`
            p_met.last_token_time = max(p_met.last_token_time,
                                        c_met.last_token_time)
            p_met.first_scheduled_time = _none_safe_min(
                p_met.first_scheduled_time, c_met.first_scheduled_time)
            p_met.first_token_time = _none_safe_min(p_met.first_token_time,
                                                    c_met.first_token_time)
            p_met.time_in_queue = _none_safe_sum(p_met.time_in_queue,
                                                 c_met.time_in_queue)
            p_met.finished_time = _none_safe_max(p_met.finished_time,
                                                 c_met.finished_time)
            p_met.last_token_time = max(p_met.last_token_time,
                                        c_met.last_token_time)
            p_met.model_execute_time = _none_safe_sum(p_met.model_execute_time,
                                                      c_met.model_execute_time)
            p_met.model_forward_time = _none_safe_sum(p_met.model_forward_time,
                                                      c_met.model_forward_time)
            p_met.scheduler_time = _none_safe_sum(p_met.scheduler_time,
                                                  c_met.scheduler_time)
            p_met.time_in_queue = _none_safe_sum(p_met.time_in_queue,
                                                 c_met.time_in_queue)
        parent_req_output.outputs.extend(child_req_output.outputs)
        parent_req_output.num_cached_tokens = _none_safe_sum(
            parent_req_output.num_cached_tokens,
            child_req_output.num_cached_tokens)

    def _maybe_aggregate_parallel_sampling_child_requests(
        self,
        request_outputs: List[RequestOutput],
    ) -> List[RequestOutput]:
        parent_req_ids_seen = set()
        agg_request_outputs: List[RequestOutput] = []
        #parent_req_id_to_idx: Dict[str,int]={}
        for req_output in request_outputs:
            possible_child_req_id = req_output.request_id
            maybe_child_req_info = (
                self._maybe_get_parallel_sampling_child_request_info(
                    possible_child_req_id))
            if maybe_child_req_info:
                # Aggregate child request into parallel sampling request output
                child_req_finished = req_output.finished
                parent_req_id = maybe_child_req_info.parent_req_id
                parent_req_info = (
                    self._get_parallel_sampling_parent_request_info(
                        parent_req_id))
                if parent_req_info.last_request_output is None:
                    # For a particular parent id, this is the first child
                    # request output we have seen in *any* step.
                    # Repurpose the child request output structure to be the
                    # parent request output structure
                    parent_req_info.last_request_output = req_output
                    last_req_output = parent_req_info.last_request_output
                    last_req_output.request_id = parent_req_id
                    last_req_output.finished = (
                        parent_req_info.
                        get_num_active_children_incr_n_finished_if_true(
                            last_req_output.finished) < 1)
                else:
                    last_req_output = parent_req_info.last_request_output
                    # Merge this child request output into the growing request
                    # output data structure associated with its parent.
                    self._merge_parallel_sampling_child_request_output_in_place(
                        last_req_output, req_output, parent_req_info)
                if parent_req_id not in parent_req_ids_seen:
                    # For a particular parent id, this is the first child
                    # request output we have seen in *this* step.
                    # Remember that a request output data structure for this
                    # particular parent request has already been appended
                    # to the output
                    agg_request_outputs.append(last_req_output)
                    parent_req_ids_seen.add(parent_req_id)
                if child_req_finished:
                    self._forget_parallel_sample_child_request(
                        possible_child_req_id)
                    # Track parent request's remaining child requests & erase
                    # parent request metadata if there are no remaining child
                    # requests
                    if self.parent_req_id_info[
                            parent_req_id].gen_num_active_children() < 1:
                        self._forget_parallel_sample_parent_request(
                            parent_req_id)
            else:
                # Not a parallel sampling request; don't touch it
                agg_request_outputs.append(req_output)

        return agg_request_outputs

    def step(self) -> List[RequestOutput]:

        # 1) Get EngineCoreOutput from the EngineCore.
        engine_core_outputs = self.engine_core.get_output()

        # 2) Detokenizer the EngineCoreOutput.
        request_outputs, requests_to_abort = self.detokenizer.step(
            engine_core_outputs)

        # 3) If necessary, aggregate outputs for parallel sampling child
        #    requests to be associated with parent request
        request_outputs = (
            self._maybe_aggregate_parallel_sampling_child_requests(
                request_outputs))

        # 4) Abort requests that finished due to stopping criteria.
        if requests_to_abort:
            self.abort_request(requests_to_abort)

        return request_outputs

    # TODO(rob): Can we get rid of these?

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
        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()
