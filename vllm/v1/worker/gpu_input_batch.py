# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Datastructures defining a GPU input batch

from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import torch
from typing_extensions import deprecated

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItems
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import length_from_prompt_token_ids_or_embeds, swap_dict_values
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.logits_processor import (BatchUpdateBuilder,
                                             LogitsProcessors,
                                             MoveDirectionality)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.utils import is_spec_decode_unsupported
from vllm.v1.utils import copy_slice
from vllm.v1.worker.block_table import MultiGroupBlockTable


@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: Optional[list[int]]
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    generator: Optional[torch.Generator]

    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]

    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[int] = None

    lora_request: Optional[LoRARequest] = None
    prompt_embeds: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    # Temporary back-compatibility for plugins that define model runner
    @property
    @deprecated("`mm_inputs` is superseded by `mm_kwargs` and will be "
                "removed in v0.13. Please use `mm_kwargs` instead.")
    def mm_inputs(self) -> list[MultiModalKwargsItems]:
        return [
            MultiModalKwargsItems.from_seq([f.data]) for f in self.mm_features
            if f.data is not None
        ]

    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            if self.prompt_token_ids is None:
                raise ValueError(
                    f"Tried to access token index {idx}, but that token was "
                    "provided via prompt_embeds, and its ID is unknown.")
            return self.prompt_token_ids[idx]
        elif idx - self.num_prompt_tokens < len(self.output_token_ids):
            return self.output_token_ids[idx - self.num_prompt_tokens]
        else:
            return -1


class InputBatch:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        logitsprocs: Optional[LogitsProcessors] = None,
        is_spec_decode: bool = False,
        is_pooling_model: bool = False,
        num_speculative_tokens: int = 0,
    ):
        self.is_pooling_model = is_pooling_model
        self.is_spec_decode = is_spec_decode
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        self._req_ids: list[Optional[str]] = []
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not
        # need to be pinned.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        self.is_token_ids = torch.zeros((max_num_reqs, max_model_len),
                                        device="cpu",
                                        dtype=bool,
                                        pin_memory=False)
        # Store prompt embeddings per request to avoid OOM from large upfront
        # allocation if max_model_len is big.
        # Maps req_index -> tensor of shape (num_prompt_tokens, hidden_size)
        self.req_prompt_embeds: dict[int, torch.Tensor] = {}
        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs, ),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_computed_tokens_cpu = \
            self.num_computed_tokens_cpu_tensor.numpy()

        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
            num_speculative_tokens=num_speculative_tokens,
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

        # IDs of requests which do not support spec decoding
        self.spec_decode_unsupported_reqs: set[str] = set()

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

        # Speculative decoding
        self.num_accepted_tokens_cpu_tensor = torch.ones((max_num_reqs, ),
                                                         dtype=torch.int64,
                                                         device="cpu",
                                                         pin_memory=pin_memory)
        self.num_accepted_tokens_cpu = \
            self.num_accepted_tokens_cpu_tensor.numpy()

        # lora related
        self.request_lora_mapping = np.zeros((self.max_num_reqs, ),
                                             dtype=np.int32)
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

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

        # Internal representation of per-step batch state changes, used for
        # reordering persistent batch and generating logitsprocs batch state
        # updates. Should reset each step.
        self.batch_update_builder = BatchUpdateBuilder()

        # TODO convert this to LogitsProcessor
        self.has_allowed_token_ids: set[str] = set()
        # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: Optional[torch.Tensor] = None
        self.allowed_token_ids_mask_cpu_tensor: Optional[torch.Tensor] = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.logits_processing_needs_token_ids = np.zeros(max_num_reqs,
                                                          dtype=bool)

        self.req_output_token_ids: list[Optional[list[int]]] = []

        # Store provided logitsprocs. If none are provided, initialize empty
        # data structure
        self.logitsprocs = logitsprocs or LogitsProcessors()

        # This is updated each time the batch constituents change.
        self.sampling_metadata = self._make_sampling_metadata()

        self.pooling_params: dict[str, PoolingParams] = {}

        # Cached reference to the GPU tensor of previously sampled tokens
        self.prev_sampled_token_ids: Optional[torch.Tensor] = None
        self.prev_sampled_token_ids_invalid_indices: Optional[set[int]] = None
        self.prev_req_id_to_index: Optional[dict[str, int]] = None

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    def _register_add_request(self, request: "CachedRequestState") -> int:
        """Track add-request operations for logits processors.
        Not applicable to pooling models.
        """

        # Fill the next empty index if there is one.
        if (new_req_index := self.batch_update_builder.pop_removed()) is None:
            # Append to end otherwise.
            new_req_index = self.num_reqs

        assert new_req_index < self.max_num_reqs
        self.batch_update_builder.batch_changed = True
        if request.sampling_params:
            # Detailed added request metadata is only required for non-pooling
            # models, to support logitsprocs.
            self.batch_update_builder.added.append(
                (new_req_index, request.sampling_params,
                 request.prompt_token_ids, request.output_token_ids))

        return new_req_index

    def add_request(
        self,
        request: "CachedRequestState",
    ) -> int:
        req_index = self._register_add_request(request)

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds)
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        if request.prompt_token_ids is not None:
            self.token_ids_cpu[
                req_index, :num_prompt_tokens] = request.prompt_token_ids
            self.is_token_ids[req_index, :num_prompt_tokens] = True
        else:
            self.is_token_ids[req_index, :num_prompt_tokens] = False
        if request.prompt_embeds is not None:
            self.req_prompt_embeds[req_index] = request.prompt_embeds
        self.token_ids_cpu[req_index,
                           start_idx:end_idx] = request.output_token_ids
        self.is_token_ids[req_index, start_idx:end_idx] = True
        # Number of token ids in prompt (token_ids_cpu or prompt_embeds).
        # NOTE(woosuk): This may include spec decode tokens.
        self.num_tokens[req_index] = request.num_tokens
        # Number of tokens without spec decode tokens.
        self.num_tokens_no_spec[req_index] = request.num_tokens

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)

        if sampling_params := request.sampling_params:
            if (self.is_spec_decode
                    and is_spec_decode_unsupported(sampling_params)):
                self.spec_decode_unsupported_reqs.add(req_id)
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
            self.frequency_penalties_cpu[
                req_index] = sampling_params.frequency_penalty
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

            # NOTE(woosuk): self.generators should not include the requests that
            # do not have their own generator.
            if request.generator is not None:
                self.generators[req_index] = request.generator

            if sampling_params.logprobs is not None:
                self.num_logprobs[req_id] = (self.vocab_size
                                             if sampling_params.logprobs == -1
                                             else sampling_params.logprobs)
            if sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.vocab_size if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs)

            if sampling_params.allowed_token_ids:
                self.has_allowed_token_ids.add(req_id)
                if self.allowed_token_ids_mask_cpu_tensor is None:
                    # Lazy allocation for this tensor, which can be large.
                    # False means we don't fill with -inf.
                    self.allowed_token_ids_mask = torch.zeros(
                        self.max_num_reqs,
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
        elif pooling_params := request.pooling_params:
            self.pooling_params[req_id] = pooling_params
            self.logits_processing_needs_token_ids[req_index] = (
                pooling_params.requires_token_ids)
        else:
            raise NotImplementedError("Unrecognized request type")

        # Speculative decoding: by default 1 token is generated.
        self.num_accepted_tokens_cpu[req_index] = 1

        # Add request lora ID
        if request.lora_request:
            lora_id = request.lora_request.lora_int_id
            if lora_id not in self.lora_id_to_request_ids:
                self.lora_id_to_request_ids[lora_id] = set()

            self.request_lora_mapping[req_index] = lora_id
            self.lora_id_to_request_ids[lora_id].add(request.req_id)
            self.lora_id_to_lora_request[lora_id] = request.lora_request
        else:
            # No LoRA
            self.request_lora_mapping[req_index] = 0

        return req_index

    def remove_request(self, req_id: str) -> Optional[int]:
        """This method must always be followed by a call to condense().

        Args:
          req_id: request to remove

        Returns:
          Removed request index, or `None` if `req_id` not recognized
        """

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None

        self.batch_update_builder.removed_append(req_index)
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        # LoRA
        lora_id = self.request_lora_mapping[req_index]
        if lora_id != 0:
            lora_req_ids = self.lora_id_to_request_ids[lora_id]
            lora_req_ids.discard(req_id)
            if not lora_req_ids:
                del self.lora_id_to_request_ids[lora_id]
                del self.lora_id_to_lora_request[lora_id]
            self.request_lora_mapping[req_index] = 0

        if self.is_pooling_model:
            self.pooling_params.pop(req_id, None)
            return req_index

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.spec_decode_unsupported_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.num_prompt_logprobs.pop(req_id, None)
        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)

        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask_cpu_tensor is not None:
            # False means we don't fill with -inf.
            self.allowed_token_ids_mask_cpu_tensor[req_index].fill_(False)
        self.bad_words_token_ids.pop(req_index, None)
        return req_index

    def swap_states(self, i1: int, i2: int) -> None:
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]
        self._req_ids[i1], self._req_ids[i2] =\
            self._req_ids[i2], self._req_ids[i1] # noqa
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] =\
            self.req_output_token_ids[i2], self.req_output_token_ids[i1]
        assert old_id_i1 is not None and old_id_i2 is not None
        self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2] =\
            self.req_id_to_index[old_id_i2], self.req_id_to_index[old_id_i1]
        self.num_tokens[i1], self.num_tokens[i2] =\
            self.num_tokens[i2], self.num_tokens[i1]
        self.num_tokens_no_spec[i1], self.num_tokens_no_spec[i2] =\
            self.num_tokens_no_spec[i2], self.num_tokens_no_spec[i1]
        self.num_prompt_tokens[i1], self.num_prompt_tokens[i2] =\
            self.num_prompt_tokens[i2], self.num_prompt_tokens[i1]
        self.num_computed_tokens_cpu[i1], self.num_computed_tokens_cpu[i2] =\
            self.num_computed_tokens_cpu[i2], self.num_computed_tokens_cpu[i1]

        # NOTE: the following is unsafe
        # self.token_ids_cpu[i1, ...], self.token_ids_cpu[i2, ...], =\
        #     self.token_ids_cpu[i2, ...], self.token_ids_cpu[i1, ...]
        # instead, we need to temporiarily copy the data for one of the indices
        # TODO(lucas): optimize this by only copying valid indices
        tmp = self.token_ids_cpu[i1, ...].copy()
        self.token_ids_cpu[i1, ...] = self.token_ids_cpu[i2, ...]
        self.token_ids_cpu[i2, ...] = tmp

        self.is_token_ids[[i1, i2], ...] = self.is_token_ids[[i2, i1], ...]

        # Swap prompt embeddings if they exist
        embeds_i1 = self.req_prompt_embeds.get(i1)
        embeds_i2 = self.req_prompt_embeds.get(i2)
        if embeds_i1 is not None:
            self.req_prompt_embeds[i2] = embeds_i1
        else:
            self.req_prompt_embeds.pop(i2, None)
        if embeds_i2 is not None:
            self.req_prompt_embeds[i1] = embeds_i2
        else:
            self.req_prompt_embeds.pop(i1, None)

        self.block_table.swap_row(i1, i2)

        self.request_lora_mapping[i1], self.request_lora_mapping[i2] = \
            self.request_lora_mapping[i2], self.request_lora_mapping[i1]

        if self.is_pooling_model:
            # Sampling and logits parameters don't apply to pooling models.
            return

        # For autoregressive models, track detailed request reordering info
        # to support logitsprocs.
        self.batch_update_builder.moved.append(
            (i1, i2, MoveDirectionality.SWAP))

        self.temperature_cpu[i1], self.temperature_cpu[i2] = \
            self.temperature_cpu[i2], self.temperature_cpu[i1]
        self.top_p_cpu[i1], self.top_p_cpu[i2] = \
            self.top_p_cpu[i2], self.top_p_cpu[i1]
        self.top_k_cpu[i1], self.top_k_cpu[i2] = \
            self.top_k_cpu[i2], self.top_k_cpu[i1]
        self.frequency_penalties_cpu[i1], self.frequency_penalties_cpu[i2] = \
            self.frequency_penalties_cpu[i2], self.frequency_penalties_cpu[i1]
        self.presence_penalties_cpu[i1], self.presence_penalties_cpu[i2] = \
            self.presence_penalties_cpu[i2], self.presence_penalties_cpu[i1]
        self.repetition_penalties_cpu[i1], self.repetition_penalties_cpu[i2] = \
            self.repetition_penalties_cpu[i2], self.repetition_penalties_cpu[i1]
        self.num_accepted_tokens_cpu[i1], self.num_accepted_tokens_cpu[i2] =\
            self.num_accepted_tokens_cpu[i2], self.num_accepted_tokens_cpu[i1]

        swap_dict_values(self.generators, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)

        if self.allowed_token_ids_mask_cpu_tensor is not None:
            self.allowed_token_ids_mask_cpu_tensor[i1], \
                self.allowed_token_ids_mask_cpu_tensor[i2] =\
                self.allowed_token_ids_mask_cpu_tensor[i2], \
                    self.allowed_token_ids_mask_cpu_tensor[i1]

    def condense(self) -> None:
        """Slide non-empty requests down into lower, empty indices.

        Any consecutive empty indices at the very end of the list are not
        filled.

        Returns:
          swaps: list of (from,to) swap tuples for moved requests
          empty_req_indices: indices not filled by condensation
        """
        num_reqs = self.num_reqs

        if not (empty_req_indices := self.batch_update_builder.removed):
            # All removed requests were replaced by added requests, or else no
            # requests were removed at all. No condense() needed
            return
        if num_reqs == 0:
            # The batched states are empty.
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = self.batch_update_builder.peek_removed()
            assert empty_index is not None
            if empty_index >= last_req_index:
                break

            # Move active request down into empty request
            # index.
            self.batch_update_builder.pop_removed()
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            num_tokens = self.num_tokens[last_req_index]
            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens]
            self.is_token_ids[empty_index, :num_tokens] = self.is_token_ids[
                last_req_index, :num_tokens]
            if last_req_index in self.req_prompt_embeds:
                self.req_prompt_embeds[
                    empty_index] = self.req_prompt_embeds.pop(last_req_index)
            self.num_tokens[empty_index] = num_tokens
            self.num_tokens_no_spec[empty_index] = self.num_tokens_no_spec[
                last_req_index]
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[
                last_req_index]
            self.num_computed_tokens_cpu[
                empty_index] = self.num_computed_tokens_cpu[last_req_index]
            self.block_table.move_row(last_req_index, empty_index)

            self.request_lora_mapping[empty_index] = self.request_lora_mapping[
                last_req_index]

            if self.is_pooling_model:
                last_req_index -= 1
                # Sampling state not used by pooling models.
                continue

            # Autoregressive models require detailed tracking of condense
            # operations to support logitsprocs
            self.batch_update_builder.moved.append(
                (last_req_index, empty_index,
                 MoveDirectionality.UNIDIRECTIONAL))

            self.temperature_cpu[empty_index] = self.temperature_cpu[
                last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.frequency_penalties_cpu[
                empty_index] = self.frequency_penalties_cpu[last_req_index]
            self.presence_penalties_cpu[
                empty_index] = self.presence_penalties_cpu[last_req_index]
            self.repetition_penalties_cpu[
                empty_index] = self.repetition_penalties_cpu[last_req_index]
            self.num_accepted_tokens_cpu[
                empty_index] = self.num_accepted_tokens_cpu[last_req_index]
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            # TODO convert these to LogitsProcessors
            if self.allowed_token_ids_mask_cpu_tensor is not None:
                self.allowed_token_ids_mask_cpu_tensor[
                    empty_index] = self.allowed_token_ids_mask_cpu_tensor[
                        last_req_index]

            bad_words_token_ids = self.bad_words_token_ids.pop(
                last_req_index, None)
            if bad_words_token_ids is not None:
                self.bad_words_token_ids[empty_index] = bad_words_token_ids

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # Trim lists to the batch size.
        del self._req_ids[num_reqs:]
        del self.req_output_token_ids[num_reqs:]

    def refresh_metadata(self):
        """Apply any batch updates to sampling metadata."""

        if self.is_pooling_model:
            batch_changed = self.batch_update_builder.reset()
            if batch_changed:
                self.sampling_metadata = self._make_sampling_metadata()
            return

        # For non-pooling models - generate and apply logitsprocs update;
        # reset batch update tracking.
        # Update sampling metadata if batch state is changed.
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
        if batch_update:
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

        needs_prompt_token_ids = (
            not self.no_penalties
            or self.logits_processing_needs_token_ids[:num_reqs].any())
        if needs_prompt_token_ids:
            # The prompt tokens are used only for applying penalties or
            # step pooling during the sampling/pooling process.
            # Hence copy these tensors only when there are requests which
            # need penalties/step_pooler to be applied.
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
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs],
            presence_penalties=self.presence_penalties[:num_reqs],
            repetition_penalties=self.repetition_penalties[:num_reqs],
            output_token_ids=cast(list[list[int]], self.req_output_token_ids),
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
            logitsprocs=self.logitsprocs,
        )

    def get_pooling_params(self) -> list[PoolingParams]:
        assert len(self.req_ids) == len(self.pooling_params)
        return [self.pooling_params[req_id] for req_id in self.req_ids]

    def get_pooling_metadata(self) -> PoolingMetadata:
        pooling_params = self.get_pooling_params()

        return PoolingMetadata(
            prompt_lens=torch.from_numpy(
                self.num_prompt_tokens[:self.num_reqs]),
            prompt_token_ids=self.sampling_metadata.prompt_token_ids,
            pooling_params=pooling_params,
        )

    def _make_prompt_token_ids_tensor(self) -> torch.Tensor:
        num_reqs = self.num_reqs
        max_prompt_len = self.num_prompt_tokens[:num_reqs].max()
        prompt_token_ids_cpu_tensor = torch.empty(
            (self.num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[:num_reqs, :max_prompt_len]
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        for i in range(num_reqs):
            prompt_token_ids[i, self.num_prompt_tokens[i]:] = self.vocab_size
        return prompt_token_ids_cpu_tensor.to(device=self.device,
                                              non_blocking=True)

    def make_lora_inputs(
        self, num_scheduled_tokens: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        """
        Given the num_scheduled_tokens for each request in the batch, return
        datastructures used to activate the current LoRAs.
        Returns:
            1. prompt_lora_mapping: A tuple of size self.num_reqs where,
               prompt_lora_mapping[i] is the LoRA id to use for the ith prompt.
            2. token_lora_mapping: A tuple of size np.sum(num_scheduled_tokens)
               where, token_lora_mapping[i] is the LoRA id to use for ith token.
            3. lora_requests: Set of relevant LoRA requests.
        """

        req_lora_mapping = self.request_lora_mapping[:self.num_reqs]
        prompt_lora_mapping = tuple(req_lora_mapping)
        token_lora_mapping = tuple(
            req_lora_mapping.repeat(num_scheduled_tokens))
        active_lora_requests: set[LoRARequest] = set(
            self.lora_id_to_lora_request.values())

        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

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
