# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Datastructures defining an input batch

from dataclasses import dataclass, field
from typing import Optional, cast

import torch

from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import swap_dict_values
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.utils import copy_slice
from vllm.v1.worker.gpu_base_input_batch import (BaseInputBatch,
                                                 BaseRequestState)

_SAMPLING_EPS = 1e-5


@dataclass
class SamplingRequestState(BaseRequestState):

    sampling_params: SamplingParams = SamplingParams()
    generator: Optional[torch.Generator] = None
    output_token_ids: list[int] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        else:
            return self.output_token_ids[idx - self.num_prompt_tokens]


class InputBatch(BaseInputBatch[SamplingRequestState]):

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],
    ):
        super().__init__(
            max_num_reqs,
            max_model_len,
            max_num_batched_tokens,
            device,
            pin_memory,
            vocab_size,
            block_sizes,
        )

        # Sampling-related.
        self.temperature = torch.empty((max_num_reqs, ),
                                       dtype=torch.float32,
                                       device=device)
        self.temperature_cpu_tensor = torch.empty((max_num_reqs, ),
                                                  dtype=torch.float32,
                                                  device="cpu",
                                                  pin_memory=pin_memory)
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)
        self.top_p_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: set[str] = set()

        self.top_k = torch.empty((max_num_reqs, ),
                                 dtype=torch.int32,
                                 device=device)
        self.top_k_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: set[str] = set()

        self.min_p = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)
        self.min_p_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()
        self.min_p_reqs: set[str] = set()

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
        self.frequency_penalties_reqs: set[str] = set()

        # Presence penalty related data structures
        self.presence_penalties = torch.empty((max_num_reqs, ),
                                              dtype=torch.float,
                                              device=device)
        self.presence_penalties_cpu_tensor = torch.empty((max_num_reqs, ),
                                                         dtype=torch.float,
                                                         device="cpu",
                                                         pin_memory=pin_memory)
        self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy(
        )
        self.presence_penalties_reqs: set[str] = set()

        # Repetition penalty related data structures
        self.repetition_penalties = torch.empty((max_num_reqs, ),
                                                dtype=torch.float,
                                                device=device)
        self.repetition_penalties_cpu_tensor = torch.empty(
            (max_num_reqs, ),
            dtype=torch.float,
            device="cpu",
            pin_memory=pin_memory)
        self.repetition_penalties_cpu = \
            self.repetition_penalties_cpu_tensor.numpy()
        self.repetition_penalties_reqs: set[str] = set()

        # req_index -> (min_tokens, stop_token_ids)
        self.min_tokens: dict[int, tuple[int, set[int]]] = {}

        # req_index -> generator
        # NOTE(woosuk): The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: dict[int, torch.Generator] = {}

        self.num_logprobs: dict[str, int] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}

        # To accumulate prompt logprobs tensor chunks across prefill steps.
        self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}

        self.logit_bias: list[Optional[dict[int,
                                            float]]] = [None] * max_num_reqs
        self.has_allowed_token_ids: set[str] = set()
        # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: Optional[torch.Tensor] = None
        self.allowed_token_ids_mask_cpu_tensor: Optional[torch.Tensor] = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.req_output_token_ids: list[Optional[list[int]]] = []

        # This is updated each time the batch constituents change.
        self.sampling_metadata = self._make_sampling_metadata()

    def add_request(
        self,
        request: "SamplingRequestState",
        req_index: Optional[int] = None,
    ) -> None:

        req_index = super()._add_request(request, req_index)

        if req_index == len(self.req_output_token_ids):
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self.req_output_token_ids[req_index] = request.output_token_ids

        req_id = request.req_id
        num_prompt_tokens = len(request.prompt_token_ids)

        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index,
                           start_idx:end_idx] = request.output_token_ids

        # Number of tokens without spec decode tokens.
        self.num_tokens_no_spec[req_index] = request.num_tokens

        sampling_params = request.sampling_params
        if sampling_params.sampling_type == SamplingType.GREEDY:
            # Avoid later division by zero.
            self.temperature_cpu[req_index] = -1.0
            self.greedy_reqs.add(req_id)
        else:
            self.temperature_cpu[req_index] = sampling_params.temperature
            self.random_reqs.add(req_id)

        self.top_p_cpu[req_index] = sampling_params.top_p
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)
        top_k = sampling_params.top_k
        if 0 < top_k < self.vocab_size:
            self.top_k_reqs.add(req_id)
        else:
            top_k = self.vocab_size
        self.top_k_cpu[req_index] = top_k
        self.min_p_cpu[req_index] = sampling_params.min_p
        self.frequency_penalties_cpu[
            req_index] = sampling_params.frequency_penalty
        if sampling_params.min_p > _SAMPLING_EPS:
            self.min_p_reqs.add(req_id)
        if sampling_params.frequency_penalty != 0.0:
            self.frequency_penalties_reqs.add(req_id)
        self.presence_penalties_cpu[
            req_index] = sampling_params.presence_penalty
        if sampling_params.presence_penalty != 0.0:
            self.presence_penalties_reqs.add(req_id)
        self.repetition_penalties_cpu[
            req_index] = sampling_params.repetition_penalty
        if sampling_params.repetition_penalty != 1.0:
            self.repetition_penalties_reqs.add(req_id)
        if sampling_params.min_tokens:
            self.min_tokens[req_index] = (sampling_params.min_tokens,
                                          sampling_params.all_stop_token_ids)

        # NOTE(woosuk): self.generators should not include the requests that
        # do not have their own generator.
        if request.generator is not None:
            self.generators[req_index] = request.generator

        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = sampling_params.logprobs
        if sampling_params.prompt_logprobs is not None:
            self.num_prompt_logprobs[req_id] = sampling_params.prompt_logprobs
        if sampling_params.logit_bias is not None:
            self.logit_bias[req_index] = sampling_params.logit_bias

        if sampling_params.allowed_token_ids:
            self.has_allowed_token_ids.add(req_id)
            if self.allowed_token_ids_mask_cpu_tensor is None:
                # Lazy allocation for this tensor, which can be large.
                # False means we don't fill with -inf.
                self.allowed_token_ids_mask = torch.zeros(self.max_num_reqs,
                                                          self.vocab_size,
                                                          dtype=torch.bool,
                                                          device=self.device)
                self.allowed_token_ids_mask_cpu_tensor = torch.zeros(
                    self.max_num_reqs,
                    self.vocab_size,
                    dtype=torch.bool,
                    device="cpu")
            self.allowed_token_ids_mask_cpu_tensor[req_index] = True
            # False means we don't fill with -inf.
            self.allowed_token_ids_mask_cpu_tensor[req_index][
                sampling_params.allowed_token_ids] = False

        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[
                req_index] = sampling_params.bad_words_token_ids

    def remove_request(self, req_id: str) -> Optional[int]:
        """This method must always be followed by a call to condense()."""

        req_index = self.req_id_to_index.get(req_id, None)
        if req_index is not None:
            self.greedy_reqs.discard(req_id)
            self.random_reqs.discard(req_id)
            self.top_p_reqs.discard(req_id)
            self.top_k_reqs.discard(req_id)
            self.min_p_reqs.discard(req_id)
            self.min_tokens.pop(req_index, None)
            self.frequency_penalties_reqs.discard(req_id)
            self.presence_penalties_reqs.discard(req_id)
            self.repetition_penalties_reqs.discard(req_id)
            self.generators.pop(req_index, None)
            self.num_logprobs.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
            self.in_progress_prompt_logprobs_cpu.pop(req_id, None)

            self.logit_bias[req_index] = None
            self.has_allowed_token_ids.discard(req_id)
            if self.allowed_token_ids_mask_cpu_tensor is not None:
                # False means we don't fill with -inf.
                self.allowed_token_ids_mask_cpu_tensor[req_index].fill_(False)
            self.bad_words_token_ids.pop(req_index, None)

        return super().remove_request(req_id)

    def swap_or_move_states(self,
                            i1: int,
                            i2: int,
                            move: bool = False) -> None:

        super().swap_or_move_states(i1, i2, move)

        if not move:
            tmp1 = (
                self.req_output_token_ids[i1],
                self.num_tokens_no_spec[i1],
                self.temperature_cpu[i1],
                self.top_p_cpu[i1],
                self.top_k_cpu[i1],
                self.frequency_penalties_cpu[i1],
                self.presence_penalties_cpu[i1],
                self.repetition_penalties_cpu[i1],
                self.min_p_cpu[i1],
                self.logit_bias[i1],
            )

        self.req_output_token_ids[i1] = self.req_output_token_ids[i2]
        self.num_tokens_no_spec[i1] = self.num_tokens_no_spec[i2]
        self.temperature_cpu[i1] = self.temperature_cpu[i2]
        self.top_p_cpu[i1] = self.top_p_cpu[i2]
        self.top_k_cpu[i1] = self.top_k_cpu[i2]
        self.frequency_penalties_cpu[i1] = self.frequency_penalties_cpu[i2]
        self.presence_penalties_cpu[i1] = self.presence_penalties_cpu[i2]
        self.repetition_penalties_cpu[i1] = self.repetition_penalties_cpu[i2]
        self.min_p_cpu[i1] = self.min_p_cpu[i2]
        self.logit_bias[i1] = self.logit_bias[i2]

        if not move:
            (
                self.req_output_token_ids[i2],
                self.num_tokens_no_spec[i2],
                self.temperature_cpu[i2],
                self.top_p_cpu[i2],
                self.top_k_cpu[i2],
                self.frequency_penalties_cpu[i2],
                self.presence_penalties_cpu[i2],
                self.repetition_penalties_cpu[i2],
                self.min_p_cpu[i2],
                self.logit_bias[i2],
            ) = tmp1
            self.req_output_token_ids[i2] = None

        swap_dict_values(self.generators, i1, i2)
        swap_dict_values(self.min_tokens, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)

        if self.allowed_token_ids_mask_cpu_tensor is not None:
            self.allowed_token_ids_mask_cpu_tensor[i1], \
                self.allowed_token_ids_mask_cpu_tensor[i2] =\
                self.allowed_token_ids_mask_cpu_tensor[i2], \
                    self.allowed_token_ids_mask_cpu_tensor[i1]

    def condense(self, empty_req_indices: list[int]) -> None:

        super().condense(empty_req_indices)
        if self.num_reqs == 0:
            self.req_output_token_ids.clear()
        else:
            del self.req_output_token_ids[self.num_reqs:]

    def refresh(self):
        self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:
        num_reqs = self.num_reqs
        if not self.all_greedy:
            temperature = copy_slice(self.temperature_cpu_tensor,
                                     self.temperature, num_reqs)
        else:
            temperature = None
        if not self.no_top_p:
            copy_slice(self.top_p_cpu_tensor, self.top_p, num_reqs)
        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)
        if not self.no_min_p:
            copy_slice(self.min_p_cpu_tensor, self.min_p, num_reqs)

        if not self.no_penalties:
            # Since syncing these tensors is expensive only copy them
            # if necessary i.e. if there are requests which require
            # penalties to be applied during sampling.
            copy_slice(self.frequency_penalties_cpu_tensor,
                       self.frequency_penalties, num_reqs)
            copy_slice(self.presence_penalties_cpu_tensor,
                       self.presence_penalties, num_reqs)
            copy_slice(self.repetition_penalties_cpu_tensor,
                       self.repetition_penalties, num_reqs)

            # The prompt tokens are used only for applying penalties during
            # the sampling process. Hence copy these tensors only when
            # there are requests which need penalties to be applied.
            prompt_token_ids = self._make_prompt_token_ids_tensor()
        else:
            prompt_token_ids = None

        allowed_token_ids_mask: Optional[torch.Tensor] = None
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            copy_slice(self.allowed_token_ids_mask_cpu_tensor,
                       self.allowed_token_ids_mask, num_reqs)
            allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p[:num_reqs],
            top_k=None if self.no_top_k else self.top_k[:num_reqs],
            min_p=None if self.no_min_p else self.min_p[:num_reqs],
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs],
            presence_penalties=self.presence_penalties[:num_reqs],
            repetition_penalties=self.repetition_penalties[:num_reqs],
            output_token_ids=cast(list[list[int]], self.req_output_token_ids),
            min_tokens=self.min_tokens,
            no_penalties=self.no_penalties,
            logit_bias=self.logit_bias[:num_reqs],
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
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
    def no_min_p(self) -> bool:
        return len(self.min_p_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        return (len(self.presence_penalties_reqs) == 0
                and len(self.frequency_penalties_reqs) == 0
                and len(self.repetition_penalties_reqs) == 0)

    @property
    def max_num_logprobs(self) -> Optional[int]:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_prompt_logprob(self) -> bool:
        return not self.num_prompt_logprobs

    @property
    def no_allowed_token_ids(self) -> bool:
        return len(self.has_allowed_token_ids) == 0
