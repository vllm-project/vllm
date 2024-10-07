import time
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Tuple, Type, TypeVar, Union)

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.outputs_v2 import SamplerOutput, ModelRunnerOutput
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import (DeviceMemoryProfiler, is_pin_memory_available, cdiv,
                        STR_DTYPE_TO_TORCH_DTYPE)
from vllm.multimodal import MultiModalDataDict
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

if TYPE_CHECKING:
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
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        self.kv_caches: List[torch.Tensor] = []

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}
        self.persistent_batch = PersistentBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
        )

        self.cum1 = 0
        self.cum2 = 0
        self.cum3 = 0

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove stopped requests from the cached states.
        # Keep the states of the pre-empted requests.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        for req_id in scheduler_output.aborted_req_ids:
            self.requests.pop(req_id, None)

        # Remove the requests from the persistent batch.
        stopped_req_ids = set().union(
            scheduler_output.preempted_req_ids,
            scheduler_output.finished_req_ids,
            scheduler_output.aborted_req_ids,
        )
        removed_req_indices: List[int] = []
        for req_id in stopped_req_ids:
            req_index = self.persistent_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Update the states of the running requests.
        start = time.time()
        for req_data in scheduler_output.scheduled_running_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]
            req_index = self.persistent_batch.req_id_to_index[req_id]

            # Update the num_computed_tokens.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            self.persistent_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)

            # Update the block table.
            num_new_blocks = len(req_data.new_block_ids)
            if num_new_blocks == 0:
                continue
            start_block_index = len(req_state.block_ids)
            req_state.block_ids.extend(req_data.new_block_ids)
            for i, block_id in enumerate(req_data.new_block_ids):
                self.persistent_batch.block_table_cpu[req_index,
                                                      start_block_index +
                                                      i] = block_id
        end = time.time()
        self.cum2 += end - start

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=req_data.prompt_token_ids,
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

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.persistent_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.persistent_batch.condense(removed_req_indices)

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.persistent_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.persistent_batch.block_table[:num_reqs].copy_(
            self.persistent_batch.block_table_cpu[:num_reqs],
            non_blocking=True)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        max_num_scheduled_tokens = 0
        for req_id in self.persistent_batch.req_ids[:num_reqs]:
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
        num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
        assert max_num_scheduled_tokens > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        indices = np.arange(num_reqs)
        req_indices = np.repeat(indices, num_scheduled_tokens)
        req_indices = torch.from_numpy(req_indices)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange_matrix = np.tile(np.arange(max_num_scheduled_tokens),
                                (num_reqs, 1))
        mask = arange_matrix < num_scheduled_tokens[:, np.newaxis]
        arange = arange_matrix[mask]
        arange = torch.from_numpy(arange)

        # Get positions.
        positions = torch.empty((total_num_scheduled_tokens, ),
                                dtype=torch.int32,
                                device="cpu",
                                pin_memory=self.pin_memory)
        torch.add(self.persistent_batch.num_computed_tokens_cpu[req_indices],
                  arange,
                  out=positions)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = positions + req_indices * self.max_model_len
        input_ids = torch.empty((total_num_scheduled_tokens, ),
                                dtype=torch.int32,
                                device="cpu",
                                pin_memory=self.pin_memory)
        torch.index_select(self.persistent_batch.token_ids_cpu.flatten(),
                           0,
                           token_indices,
                           out=input_ids)

        # Calculate the slot mapping.
        block_numbers = self.persistent_batch.block_table_cpu.flatten()[
            token_indices // self.block_size]
        block_offsets = token_indices % self.block_size
        slot_mapping = torch.empty((total_num_scheduled_tokens, ),
                                   dtype=torch.int32,
                                   device="cpu",
                                   pin_memory=self.pin_memory)
        torch.add(block_numbers * self.block_size,
                  block_offsets,
                  out=slot_mapping)

        # Prepare the attention metadata.
        num_scheduled_tokens = torch.from_numpy(num_scheduled_tokens)
        query_start_loc = torch.empty((num_reqs + 1, ),
                                      dtype=torch.int32,
                                      device="cpu",
                                      pin_memory=self.pin_memory)
        query_start_loc[0] = 0
        torch.cumsum(num_scheduled_tokens, dim=0, out=query_start_loc[1:])

        seq_lens = (self.persistent_batch.num_computed_tokens_cpu[:num_reqs] +
                    num_scheduled_tokens)
        max_seq_len = seq_lens.max().item()
        seq_start_loc = torch.empty((num_reqs + 1, ),
                                    dtype=torch.int32,
                                    device="cpu",
                                    pin_memory=self.pin_memory)
        seq_start_loc[0] = 0
        torch.cumsum(seq_lens, dim=0, out=seq_start_loc[1:])

        # Move the tensors to the device.
        input_ids = input_ids.to(self.device, non_blocking=True)
        positions = positions.to(self.device, non_blocking=True).long()
        query_start_loc = query_start_loc.to(self.device, non_blocking=True)
        seq_start_loc = seq_start_loc.to(self.device, non_blocking=True)
        slot_mapping = slot_mapping.to(self.device, non_blocking=True).long()
        attn_metadata = FlashAttentionMetadata(
            max_query_len=max_num_scheduled_tokens,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_start_loc=seq_start_loc,
            block_table=self.persistent_batch.block_table[:num_reqs],
            slot_mapping=slot_mapping,
        )
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicty. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        return input_ids, positions, attn_metadata, logits_indices

    def _prepare_sampling(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> SamplingMetadata:
        skip_copy = True
        if (scheduler_output.aborted_req_ids
                or scheduler_output.finished_req_ids
                or scheduler_output.preempted_req_ids):
            skip_copy = False
        if (scheduler_output.scheduled_new_reqs
                or scheduler_output.scheduled_resumed_reqs):
            skip_copy = False
        # Create the sampling metadata.
        sampling_metadata = self.persistent_batch.make_sampling_metadata(
            skip_copy)
        return sampling_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        start = time.time()
        self._update_states(scheduler_output)
        end = time.time()
        self.cum1 += end - start

        start = time.time()
        inputs = self._prepare_inputs(scheduler_output)
        input_ids, positions, attn_metadata, logits_indices = inputs
        end = time.time()
        # self.cum2 += end - start

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=self.kv_caches,
            attn_metadata=attn_metadata,
        )
        hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(hidden_states)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self._prepare_sampling(scheduler_output)
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        # NOTE: CPU-GPU synchronization happens here.
        sampled_token_ids = sampler_output.sampled_token_ids.cpu()
        start = time.time()
        sampled_token_ids_list = sampled_token_ids.tolist()
        # TODO: Optimize.
        num_reqs = self.persistent_batch.num_reqs
        for i, req_id in enumerate(self.persistent_batch.req_ids[:num_reqs]):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            num_prompt_tokens = len(req_state.prompt_token_ids)
            if seq_len >= num_prompt_tokens:
                # Append the sampled token to the output token ids.
                token_id = sampled_token_ids_list[i]
                self.persistent_batch.token_ids_cpu[i, seq_len] = token_id
                req_state.output_token_ids.append(token_id)
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.persistent_batch.generators[i]
                if generator is not None:
                    offset = generator.get_offset()
                    generator = generator.set_offset(offset - 1)
                    self.persistent_batch.generators[i] = generator
        end = time.time()
        self.cum3 += end - start

        if sampler_output.logprob_token_ids is None:
            logprob_token_ids = None
        else:
            logprob_token_ids = sampler_output.logprob_token_ids.cpu()
        if sampler_output.logprobs is None:
            logprobs = None
        else:
            logprobs = sampler_output.logprobs.cpu()
        model_runner_output = ModelRunnerOutput(
            req_ids=self.persistent_batch.req_ids[:num_reqs],
            req_id_to_index=self.persistent_batch.req_id_to_index,
            sampled_token_ids_cpu=sampled_token_ids,
            logprob_token_ids_cpu=logprob_token_ids,
            logprobs_cpu=logprobs,
        )
        # print(f"cum1: {self.cum1 * 1000:.3f} ms")
        # print(f"cum2: {self.cum2 * 1000:.3f} ms")
        # print(f"cum3: {self.cum3 * 1000:.3f} ms")
        return model_runner_output

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
        # FIXME
        hidden_size = self.model_config.get_hidden_size()
        intermediate_size = int(hidden_size * 3.5)
        tp_size = self.parallel_config.tensor_parallel_size
        d = max(6 * hidden_size, 4 * intermediate_size // tp_size)
        tmp = torch.empty((self.max_num_tokens, d),
                          dtype=self.dtype,
                          device=self.device)
        return

    def capture_model(self) -> None:
        return

    def initialize_kv_cache(self, num_blocks: int) -> None:
        assert len(self.kv_caches) == 0
        # Models like Jamba, have mixed typed layers, E.g Mamba
        num_attn_layers = self.model_config.get_num_attention_layers(
            self.parallel_config)
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        for _ in range(num_attn_layers):
            kv_cache_shape = (2, num_blocks, self.block_size, num_kv_heads,
                              head_size)
            self.kv_caches.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device))


@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    multi_modal_data: Optional["MultiModalDataDict"]
    sampling_params: SamplingParams
    generator: Optional[torch.Generator]

    block_ids: List[int]
    num_computed_tokens: int
    output_token_ids: List[int]


class PersistentBatch:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        device: torch.device,
        pin_memory: bool,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.device = device
        self.pin_memory = pin_memory

        self.req_ids: List[Optional[str]] = [None] * max_num_reqs
        self.req_id_to_index: Dict[str, int] = {}

        self.token_ids_cpu = torch.empty((max_num_reqs, max_model_len),
                                         dtype=torch.int32,
                                         device="cpu")
        self.num_computed_tokens_cpu = torch.empty((max_num_reqs, ),
                                                   dtype=torch.int32,
                                                   device="cpu")

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
        request: "CachedRequestState",
        req_index: Optional[int] = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        self.req_ids[req_index] = request.req_id
        self.req_id_to_index[request.req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.token_ids_cpu[req_index, :num_prompt_tokens] = torch.as_tensor(
            request.prompt_token_ids, dtype=torch.int32, device="cpu")
        for i, token_id in enumerate(request.output_token_ids):
            self.token_ids_cpu[req_index, num_prompt_tokens + i] = token_id

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        # TODO: Optimize.
        for i, block_id in enumerate(request.block_ids):
            self.block_table_cpu[req_index, i] = block_id

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
        if sampling_params.prompt_logprobs:
            self.prompt_logprob_reqs.add(req_index)

    def remove_request(self, req_id: str) -> Optional[int]:
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self.req_ids[req_index] = None

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.generators[req_index] = None
        self.num_logprobs.pop(req_id, None)
        self.prompt_logprob_reqs.discard(req_id)
        return req_index

    def clear(self) -> None:
        self.req_ids = [None] * self.max_num_reqs
        self.req_id_to_index.clear()
        self.greedy_reqs.clear()
        self.random_reqs.clear()
        self.top_p_reqs.clear()
        self.top_k_reqs.clear()
        self.generators.clear()
        self.num_logprobs.clear()
        self.prompt_logprob_reqs.clear()

    def condense(self, empty_req_indices: List[int]) -> None:
        if self.num_reqs == 0:
            # The batched states are empty.
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
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
            req_id = self.req_ids[last_req_index]
            self.req_ids[empty_index] = req_id
            self.req_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            # TODO(woosuk): Optimize the copy of token_ids_cpu and
            # block_table_cpu.
            self.token_ids_cpu[empty_index] = self.token_ids_cpu[
                last_req_index]
            self.num_computed_tokens_cpu[
                empty_index] = self.num_computed_tokens_cpu[last_req_index]
            self.block_table_cpu[empty_index] = self.block_table_cpu[
                last_req_index]
            self.temperature_cpu[empty_index] = self.temperature_cpu[
                last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.generators[empty_index] = self.generators[last_req_index]

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

    def make_sampling_metadata(
        self,
        skip_copy: bool = False,
    ) -> SamplingMetadata:
        if not skip_copy:
            self.temperature[:self.num_reqs].copy_(
                self.temperature_cpu[:self.num_reqs], non_blocking=True)
            self.top_p[:self.num_reqs].copy_(self.top_p_cpu[:self.num_reqs],
                                             non_blocking=True)
            self.top_k[:self.num_reqs].copy_(self.top_k_cpu[:self.num_reqs],
                                             non_blocking=True)
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

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

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
        if self.num_logprobs:
            return max(self.num_logprobs.values())
        else:
            return 0

    @property
    def no_logprob(self) -> bool:
        return len(self.num_logprobs) == 0

    @property
    def no_prompt_logprob(self) -> bool:
        return len(self.prompt_logprob_reqs) == 0
