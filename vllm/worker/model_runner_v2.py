from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Tuple, Type, TypeVar, Union)

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.attention.backends.abstract import AttentionState
from vllm.attention.backends.utils import CommonAttentionState
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig)
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import (DeviceMemoryProfiler, is_pin_memory_available)
from vllm.worker.model_runner_base import dump_input_when_exception
from vllm.multimodal import MultiModalDataDict

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend
    from vllm.core.scheduler_v2 import SchedulerOutput

logger = init_logger(__name__)


class GPUModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        kv_cache_dtype: Optional[str] = "auto",
        lora_config: Optional[LoRAConfig] = None,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        observability_config: Optional[ObservabilityConfig] = None,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.prompt_adapter_config = prompt_adapter_config
        self.observability_config = observability_config

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_num_blocks_per_req = (
            (self.model_config.max_model_len + self.block_size - 1) //
            self.block_size)

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        self.kv_caches: List[torch.Tensor] = []

        # Request states.
        self.requests: Dict[str, RequestState] = {}
        self.batched_states = BatchedRequestStates(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove stopped requests from the cached states.
        # Keep the states of the pre-empted requests.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        for req_id in scheduler_output.aborted_req_ids:
            self.requests.pop(req_id, None)

        # Remove the requests from the batched states.
        stopped_req_ids = (scheduler_output.preempted_req_ids +
                           scheduler_output.finished_req_ids +
                           scheduler_output.aborted_req_ids)
        removed_req_indices: List[int] = []
        for req_id in stopped_req_ids:
            req_index = self.batched_states.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Condense the batched states.
        # We condense the states before adding new/resumed requests
        # because the attention backend may require it.
        self.batched_states.condense(removed_req_indices)

        # Update the states of the running requests.
        num_prev_blocks: Dict[str, int] = {}
        new_block_ids: Dict[str, List[int]] = {}
        for req_data in scheduler_output.scheduled_running_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            num_prev_blocks[req_id] = len(req_state.block_ids)
            new_block_ids[req_id] = req_data.new_block_ids
            req_state.block_ids.extend(req_data.new_block_ids)
            req_state.num_computed_tokens = req_data.num_computed
        # Update the block table and the number of computed tokens
        # of the running requests.
        for req_id in self.batched_states.req_ids:
            if req_id is None:
                continue
            start_block_index = num_prev_blocks[req_id]
            block_ids = new_block_ids[req_id]
            end_block_index = start_block_index + len(block_ids)
            self.batched_states.block_table_cpu[
                req_index, start_block_index:end_block_index] = block_ids
            self.batched_states.num_computed_tokens_cpu[req_index] = (
                self.requests[req_id].num_computed_tokens)

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            prompt_token_ids_cpu = torch.tensor(req_data.prompt_token_ids,
                                                device="cpu",
                                                pin_memory=self.pin_memory)
            prompt_token_ids = prompt_token_ids_cpu.to(self.device,
                                                       non_blocking=True)

            self.requests[req_id] = RequestState(
                req_id=req_id,
                prompt_token_ids=prompt_token_ids,
                prompt_token_ids_cpu=prompt_token_ids_cpu,
                prompt=req_data.prompt,
                multi_modal_data=req_data.multi_modal_data,
                sampling_params=req_data.sampling_params,
                generator=None,  # TODO
                block_ids=req_data.block_ids,
                num_computed_tokens=req_data.num_computed_tokens,
                output_token_ids=[],
            )
            req_ids_to_add.append(req_id)

        # Update the cached states of the resumed requests.
        for req_data in scheduler_output.scheduled_resumed_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            req_state.block_ids = req_data.block_ids
            req_state.num_computed_tokens = req_data.num_computed_tokens
            req_ids_to_add.append(req_id)

        # Add the new or resumed requests to the batched states.
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.batched_states.add_request(req_state)

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        assert scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

    @torch.inference_mode()
    @dump_input_when_exception(exclude_args=[0], exclude_kwargs=["self"])
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> SamplerOutput:
        self._update_states(scheduler_output)
        inputs = self._prepare_inputs(scheduler_output)
        input_ids, position_ids, attn_metadata = inputs
        # Create the sampling metadata.
        sampling_metadata = self.batched_states.get_sampling_metadata()
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            kv_caches=self.kv_caches,
        )

        logits = self.model.compute_logits(hidden_states, sampling_metadata)
        # Sample the next token and get logprobs if needed.
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return sampler_output

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:
            self.model = get_model(model_config=self.model_config,
                                   device_config=self.device_config,
                                   load_config=self.load_config,
                                   lora_config=self.lora_config,
                                   parallel_config=self.parallel_config,
                                   scheduler_config=self.scheduler_config,
                                   cache_config=self.cache_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    @torch.inference_mode()
    def profile_run(self) -> None:
        return

    def initialize_kv_cache(self) -> None:
        ...


class BatchedRequestStates:

    def __init__(
        self,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        device: torch.device,
        pin_memory: bool,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.device = device
        self.pin_memory = pin_memory

        self.num_reqs = 0
        self.req_ids: List[Optional[str]] = [None] * max_num_reqs

        self.num_computed_tokens = torch.empty((max_num_reqs, ),
                                               dtype=torch.int32,
                                               device=device)
        self.num_computed_tokens_cpu = torch.empty((max_num_reqs, ),
                                                   dtype=torch.int32,
                                                   device="cpu",
                                                   pin_memory=pin_memory)

        # Attention-related.
        self.block_table = torch.empty((max_num_reqs, max_num_blocks_per_req),
                                       device=self.device,
                                       dtype=torch.int32)
        self.block_table_cpu = torch.empty(
            (max_num_reqs, max_num_blocks_per_req),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory)

        # Sampling-related.
        self.temperature = torch.empty((max_num_reqs, ),
                                       dtype=torch.float32,
                                       device=device)
        self.temperature_cpu = torch.empty((max_num_reqs, ),
                                           dtype=torch.float32,
                                           device="cpu",
                                           pin_memory=pin_memory)
        self.greedy_reqs: Set[str] = set()
        self.random_reqs: Set[str] = set()

        self.top_p = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)
        self.top_p_cpu = torch.empty((max_num_reqs, ),
                                     dtype=torch.float32,
                                     device="cpu",
                                     pin_memory=pin_memory)
        self.top_p_reqs: Set[str] = set()

        self.top_k = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)
        self.top_k_cpu = torch.empty((max_num_reqs, ),
                                     dtype=torch.float32,
                                     device="cpu",
                                     pin_memory=pin_memory)
        self.top_k_reqs: Set[str] = set()

        self.generators: List[Optional[torch.Generator]] = [None
                                                            ] * max_num_reqs

        self.num_logprobs: Dict[str, int] = {}
        self.prompt_logprob_reqs: Set[str] = set()

    def add_request(
        self,
        request: "RequestState",
        req_index: Optional[int] = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        self.req_ids[req_index] = request.req_id
        self.num_reqs += 1

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table_cpu[
            req_index, :len(request.block_ids)] = request.block_ids

        sampling_params = request.sampling_params
        self.temperature_cpu[req_index] = sampling_params.temperature
        if sampling_params.sampling_type == SamplingType.GREEDY:
            self.greedy_reqs.add(req_index)
        elif sampling_params.sampling_type == SamplingType.RANDOM:
            self.random_reqs.add(req_index)
        elif sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            # TODO
            assert False

        self.top_p_cpu[req_index] = sampling_params.top_p
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_index)
        self.top_k_cpu[req_index] = sampling_params.top_k
        if sampling_params.top_k > 0:
            self.top_k_reqs.add(req_index)

        self.generators[req_index] = request.generator

        num_logprobs = sampling_params.logprobs
        if num_logprobs is not None and num_logprobs > 0:
            self.num_logprobs[request.req_id] = num_logprobs
        if sampling_params.prompt_logprob:
            self.prompt_logprob_reqs.add(req_index)

    def remove_request(self, req_id: str) -> Optional[int]:
        if not req_id in self.req_ids:
            return None
        req_index = self.req_ids.index(req_id)
        self.req_ids[req_index] = None
        self.num_reqs -= 1

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.generators[req_index] = None
        self.num_logprobs.pop(req_id, None)
        self.prompt_logprob_reqs.discard(req_id)
        return req_index

    def clear(self) -> None:
        self.num_reqs = 0
        self.greedy_reqs.clear()
        self.random_reqs.clear()
        self.top_p_reqs.clear()
        self.top_k_reqs.clear()
        self.generators.clear()
        self.num_logprobs.clear()
        self.prompt_logprob_reqs.clear()

    def condense(self, empty_req_indices: List[int]) -> None:
        # TODO(woosuk): Consider LoRA.
        if not empty_req_indices:
            # The batched states are already condensed.
            return
        if self.num_reqs == 0:
            # The batched states are empty.
            return

        empty_req_indices = sorted(empty_req_indices, reverse=True)
        last_req_index = self.num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = empty_req_indices.pop() 
            if empty_index >= last_req_index:
                break

            # Swap the states.
            self.req_ids[empty_index] = self.req_ids[last_req_index]
            self.num_computed_tokens_cpu[
                empty_index] = self.num_computed_tokens_cpu[last_req_index]
            self.block_table_cpu[empty_index] = self.block_table_cpu[
                last_req_index]
            self.temperature_cpu[empty_index] = self.temperature_cpu[
                last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.generators[empty_index] = self.generators[last_req_index]

    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def no_generator(self) -> bool:
        return len(self.generators) == 0

    @property
    def max_num_logprobs(self) -> int:
        return max(self.num_logprobs.values())

    @property
    def no_logprob(self) -> bool:
        return len(self.num_logprobs) == 0

    @property
    def no_prompt_logprob(self) -> bool:
        return len(self.prompt_logprob_reqs) == 0

    def get_sampling_metadata(self) -> SamplingMetadata:
        return SamplingMetadata(
            temperature=self.temperature[:self.num_reqs],
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=self.top_p[:self.num_reqs],
            top_k=self.top_k[:self.num_reqs],
            no_top_p=self.no_top_p,
            no_top_k=self.no_top_k,
            generators=self.generators[:self.num_reqs],
            no_generator=self.no_generator,
            max_num_logprobs=self.max_num_logprobs,
        )


@dataclass
class RequestState:

    req_id: str
    prompt_token_ids: torch.Tensor
    prompt_token_ids_cpu: torch.Tensor
    prompt: Optional[str]
    multi_modal_data: Optional["MultiModalDataDict"]
    sampling_params: SamplingParams
    generator: Optional[torch.Generator]

    block_ids: List[int]
    num_computed_tokens: int
    output_token_ids: List[int]
