# Datastructures defining an input batch

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import numpy as np
import torch

from vllm.multimodal import MultiModalKwargs
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.sample.metadata import SamplingMetadata

if TYPE_CHECKING:
    from vllm.multimodal.inputs import PlaceholderRange


@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    mm_inputs: List[MultiModalKwargs]
    mm_positions: List["PlaceholderRange"]
    sampling_params: SamplingParams
    generator: Optional[torch.Generator]

    block_ids: List[int]
    num_computed_tokens: int
    output_token_ids: List[int]

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


class InputBatch:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        self.req_ids: List[Optional[str]] = [None] * max_num_reqs
        self.req_id_to_index: Dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        self.num_computed_tokens_cpu = np.empty(max_num_reqs, dtype=np.int32)
        self.num_prompt_token_ids = np.empty(max_num_reqs, dtype=np.int32)

        # Attention-related.
        self.block_table = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device=self.device,
            dtype=torch.int32,
        )
        self.block_table_cpu_tensor = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.block_table_cpu = self.block_table_cpu_tensor.numpy()

        # Sampling-related.
        self.temperature = torch.empty((max_num_reqs, ),
                                       dtype=torch.float32,
                                       device=device)
        self.temperature_cpu_tensor = torch.empty((max_num_reqs, ),
                                                  dtype=torch.float32,
                                                  device="cpu",
                                                  pin_memory=pin_memory)
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: Set[str] = set()
        self.random_reqs: Set[str] = set()

        self.top_p = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)
        self.top_p_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: Set[str] = set()

        self.top_k = torch.empty((max_num_reqs, ),
                                 dtype=torch.int32,
                                 device=device)
        self.top_k_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: Set[str] = set()

        # Frequency penalty related data structures
        self.frequency_penalties = torch.empty((max_num_reqs, ),
                                               dtype=torch.float,
                                               device=device)
        self.frequency_penalties_cpu_tensor = torch.empty(
            (max_num_reqs, ),
            dtype=torch.float,
            device="cpu",
            pin_memory=pin_memory)
        self.frequency_penalties_cpu = \
            self.frequency_penalties_cpu_tensor.numpy()
        self.frequency_penalties_reqs: Set[str] = set()

        # Presence penalty related data structures
        self.presence_penalties = torch.empty((max_num_reqs, ),
                                              dtype=torch.float,
                                              device=device)
        self.presence_penalties_cpu_tensor = torch.empty((max_num_reqs, ),
                                                         dtype=torch.float,
                                                         device="cpu",
                                                         pin_memory=pin_memory)
        self.presence_penalties_cpu = \
            self.presence_penalties_cpu_tensor.numpy()
        self.presence_penalties_reqs: Set[str] = set()

        # Repetition penalty related data structures
        self.repetition_penalties = torch.empty((max_num_reqs, ),
                                                dtype=torch.float,
                                                device=device)
        self.repetition_penalties_cpu_tensor = torch.empty(
            (max_num_reqs, ),
            dtype=torch.float,
            device="cpu",
            pin_memory=pin_memory)
        self.repetition_penalties_cpu =\
            self.repetition_penalties_cpu_tensor.numpy()
        self.repetition_penalties_reqs: Set[str] = set()

        self.prompt_tokens_tensor: Optional[torch.Tensor] = None
        self.min_tokens: List[int] = [0] * max_num_reqs
        self.stop_token_ids: List[Set[int]] = [
            set() for _ in range(max_num_reqs)
        ]

        # req_index -> generator
        # NOTE(woosuk): The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: Dict[int, torch.Generator] = {}

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

        req_id = request.req_id
        self.req_ids[req_index] = req_id
        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_token_ids[req_index] = num_prompt_tokens
        self.token_ids_cpu[
            req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index,
                           start_idx:end_idx] = request.output_token_ids

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        num_blocks = len(request.block_ids)
        self.block_table_cpu[req_index, :num_blocks] = request.block_ids

        sampling_params = request.sampling_params
        self.temperature_cpu[req_index] = sampling_params.temperature
        if sampling_params.sampling_type == SamplingType.GREEDY:
            self.greedy_reqs.add(req_id)
        else:
            self.random_reqs.add(req_id)

        self.top_p_cpu[req_index] = sampling_params.top_p
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)
        self.top_k_cpu[req_index] = sampling_params.top_k
        if sampling_params.top_k > 0:
            self.top_k_reqs.add(req_id)
        self.frequency_penalties_cpu[req_index] =\
            sampling_params.frequency_penalty
        if sampling_params.frequency_penalty != 0.0:
            self.frequency_penalties_reqs.add(req_id)
        self.presence_penalties_cpu[req_index] = \
            sampling_params.presence_penalty
        if sampling_params.presence_penalty != 0.0:
            self.presence_penalties_reqs.add(req_id)
        self.repetition_penalties_cpu[req_index] = \
            sampling_params.repetition_penalty
        if sampling_params.repetition_penalty != 1.0:
            self.repetition_penalties_reqs.add(req_id)
        self.min_tokens[req_index] = sampling_params.min_tokens
        self.stop_token_ids[req_index] = sampling_params.all_stop_token_ids

        # NOTE(woosuk): self.generators should not include the requests that
        # do not have their own generator.
        if request.generator is not None:
            self.generators[req_index] = request.generator

        num_logprobs = sampling_params.logprobs
        if num_logprobs is not None and num_logprobs > 0:
            self.num_logprobs[req_id] = num_logprobs
        if sampling_params.prompt_logprobs:
            self.prompt_logprob_reqs.add(req_id)

    def remove_request(self, req_id: str) -> Optional[int]:
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self.req_ids[req_index] = None

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
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
        self.frequency_penalties_reqs.clear()
        self.presence_penalties_reqs.clear()
        self.repetition_penalties_reqs.clear()
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
            assert req_id is not None
            self.req_ids[empty_index] = req_id
            self.req_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            # TODO(woosuk): Optimize the copy of token_ids_cpu and
            # block_table_cpu.
            self.token_ids_cpu[empty_index] = self.token_ids_cpu[
                last_req_index]
            self.num_prompt_token_ids[empty_index] =\
                self.num_prompt_token_ids[last_req_index]
            self.num_computed_tokens_cpu[
                empty_index] = self.num_computed_tokens_cpu[last_req_index]
            self.block_table_cpu[empty_index] = self.block_table_cpu[
                last_req_index]
            self.temperature_cpu[empty_index] = self.temperature_cpu[
                last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.frequency_penalties_cpu[empty_index] = \
                self.frequency_penalties_cpu[last_req_index]
            self.presence_penalties_cpu[empty_index] = \
                self.presence_penalties_cpu[last_req_index]
            self.repetition_penalties_cpu[empty_index] = \
                self.repetition_penalties_cpu[last_req_index]
            self.min_tokens[empty_index] = self.min_tokens[last_req_index]
            self.stop_token_ids[empty_index] = \
                self.stop_token_ids[last_req_index]
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

    def make_sampling_metadata(
        self,
        req_id_output_token_ids: Dict[str, List[int]],
        skip_copy: bool = False,
    ) -> SamplingMetadata:
        if not skip_copy:
            self.temperature[:self.num_reqs].copy_(
                self.temperature_cpu_tensor[:self.num_reqs], non_blocking=True)
            self.top_p[:self.num_reqs].copy_(
                self.top_p_cpu_tensor[:self.num_reqs], non_blocking=True)
            self.top_k[:self.num_reqs].copy_(
                self.top_k_cpu_tensor[:self.num_reqs], non_blocking=True)
            if not self.no_penalties:
                # Since syncing these tensors is expensive only copy them
                # if necessary i.e. if there are requests which require
                # penalties to be applied during sampling.
                self.frequency_penalties[:self.num_reqs].copy_(
                    self.frequency_penalties_cpu_tensor[:self.num_reqs],
                    non_blocking=True)
                self.presence_penalties[:self.num_reqs].copy_(
                    self.presence_penalties_cpu_tensor[:self.num_reqs],
                    non_blocking=True)
                self.repetition_penalties[:self.num_reqs].copy_(
                    self.repetition_penalties_cpu_tensor[:self.num_reqs],
                    non_blocking=True)
                # The prompt tokens are used only for applying penalties during
                # the sampling process. Hence copy these tensors only when
                # there are requests which need penalties to be applied.
                self.prompt_tokens_tensor = \
                    self._construct_prompt_tokens_tensor(
                        self.vocab_size, device=self.device)

        output_token_ids: List[List[int]] = []

        for req_id in self.req_ids[:self.num_reqs]:
            assert req_id is not None
            # Currently we create a tensor for output_token_ids from scratch
            # at each step. However, for the penalties computation what we
            # need is stats about the token ids present in the output. This
            # stats can be maintained incrementally instead of computing it
            # from scratch at each step.
            # TODO - Replace this with incremental update to output token
            # statistics.
            output_token_ids.append(req_id_output_token_ids[req_id])

        return SamplingMetadata(
            temperature=self.temperature[:self.num_reqs],
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=self.top_p[:self.num_reqs],
            top_k=self.top_k[:self.num_reqs],
            no_top_p=self.no_top_p,
            no_top_k=self.no_top_k,
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=self.prompt_tokens_tensor[:self.num_reqs] \
                if self.prompt_tokens_tensor is not None else None,
            frequency_penalties=self.frequency_penalties[:self.num_reqs],
            presence_penalties=self.presence_penalties[:self.num_reqs],
            repetition_penalties=self.repetition_penalties[:self.num_reqs],
            output_token_ids=output_token_ids,
            min_tokens=self.min_tokens[:self.num_reqs],
            stop_token_ids=self.stop_token_ids[:self.num_reqs],
            no_penalties=self.no_penalties,
        )

    def _construct_prompt_tokens_tensor(
        self,
        vocab_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        max_prompt_len = max(self.num_prompt_token_ids[:self.num_reqs])
        # use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        # TODO - Add a method in vllm/utils.py to pad a numpy array similar
        # to make_tensor_with_pad which takes a list and move the logic
        # there.
        
        padded_prompts = np.full((self.num_reqs, max_prompt_len),
                                 vocab_size,
                                 dtype=np.int64)
        for i in range(self.num_reqs):
            padded_prompts[i, :self.num_prompt_token_ids[i]] =\
                self.token_ids_cpu[i, :self.num_prompt_token_ids[i]]
        prompt_tokens_cpu_tensor = torch.from_numpy(padded_prompts).to("cpu")
        if self.pin_memory:
            prompt_tokens_cpu_tensor = \
                prompt_tokens_cpu_tensor.pin_memory()
        prompt_tokens_tensor = prompt_tokens_cpu_tensor.to(device=device,
                                                           non_blocking=True)
        return prompt_tokens_tensor

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
    def no_penalties(self) -> bool:
        return len(self.presence_penalties_reqs) == 0 and \
            len(self.frequency_penalties_reqs) == 0 and \
                len(self.repetition_penalties_reqs) == 0

    @property
    def max_num_logprobs(self) -> int:
        return max(self.num_logprobs.values()) if self.num_logprobs else 0

    @property
    def no_logprob(self) -> bool:
        return len(self.num_logprobs) == 0

    @property
    def no_prompt_logprob(self) -> bool:
        return len(self.prompt_logprob_reqs) == 0
