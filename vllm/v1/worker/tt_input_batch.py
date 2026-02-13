# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, cast

import numpy as np
import torch

from vllm.v1.sample.logits_processor import (BatchUpdateBuilder,
                                             MoveDirectionality,
                                             init_builtin_logitsprocs)
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState

# Sentinel value for None seed. vLLM treats -1 as equivalent to None
# (see SamplingParams.__post_init__), so we use -1 as the sentinel.
SEED_NONE_SENTINEL = -1


class SamplingInputBatch:
    # Default values for padding sampling parameters in decode mode.
    DEFAULTS = {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": SEED_NONE_SENTINEL,  # Sentinel represents None (no seed)
        "num_logprobs": 0,  # Default to no logprobs
    }

    def __init__(self, max_num_reqs: int):
        self.max_num_reqs = max_num_reqs
        # Initialize sampling parameter tensors with default values.
        default_tensors = self.create_default_tensors()
        # Set attributes explicitly for each parameter.
        self.temperature = default_tensors["temperature"]
        self.top_p = default_tensors["top_p"]
        self.top_k = default_tensors["top_k"]
        self.presence_penalty = default_tensors["presence_penalty"]
        self.frequency_penalty = default_tensors["frequency_penalty"]
        self.repetition_penalty = default_tensors["repetition_penalty"]
        self.seed = default_tensors["seed"]
        self.num_logprobs = default_tensors["num_logprobs"]
        # Asserting that all defaults have corresponding attributes.
        for name in self.DEFAULTS:
            assert hasattr(
                self,
                name), (f"Missing attribute '{name}' in SamplingInputBatch")
        self.sampling_param_names = list(self.DEFAULTS.keys())

        # req_index -> generator
        # NOTE: The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: dict[int, torch.Generator] = {}

        # Internal representation of per-step batch state changes, used for
        # reordering persistent batch and generating logitsprocs batch state
        # updates. Should reset each step.
        self.batch_update_builder = BatchUpdateBuilder()

        # Define logits processors.
        self.logitsprocs = init_builtin_logitsprocs(
            pin_memory_available=False,
            max_num_reqs=max_num_reqs +
            1,  # not sure why but match gpu_input_batch
            device=torch.device("cpu"))

        # Allowed token IDs tracking
        self.has_allowed_token_ids: set[str] = set()
        # NOTE: In the mask tensor, if the corresponding token is allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: Optional[torch.Tensor] = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

    def pad_with_defaults(self, num_reqs: int) -> None:
        """Pad sampling parameters with default values for indices >=
        num_reqs."""
        for name in self.sampling_param_names:
            param_tensor = getattr(self, name)
            default_value = self.DEFAULTS[name]
            param_tensor[num_reqs:] = default_value

    def create_default_tensors(self) -> dict[str, torch.Tensor]:
        """Create tensors filled with default values for all parameters in
        DEFAULTS."""
        # Map Python types to PyTorch dtypes
        # Note: torch.full infers dtype, but int defaults to int64, so we
        # explicitly specify int32.
        dtype_map = {
            float: torch.float32,
            int: torch.int32,
            bool: torch.bool,
        }
        result: dict[str, torch.Tensor] = {}
        for name, default_value in self.DEFAULTS.items():
            dtype = dtype_map[type(default_value)]
            result[name] = torch.full((self.max_num_reqs, ),
                                      default_value,
                                      dtype=dtype)
        return result


class InputBatch:
    """Persistent input batch, based on InputBatch for GPU/TPU backends."""

    def __init__(
            self,
            max_num_reqs: int,
            max_model_len: int,
            max_num_batched_tokens: int,
            vocab_size: int,
            block_sizes: list[int],  # The block_size of each kv cache group
    ):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        self._req_ids: list[Optional[str]] = []
        self.req_id_to_index: dict[str, int] = {}
        # Sampling fast-path bookkeeping (track by req_id like GPUInputBatch).
        # These are used to answer common "batch-wide" queries in O(1).
        self.random_reqs: set[str] = set()
        self.presence_penalties_reqs: set[str] = set()
        self.frequency_penalties_reqs: set[str] = set()
        self.repetition_penalties_reqs: set[str] = set()

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            dtype=torch.int32,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()

        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)

        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=False,
            device="cpu",
            block_sizes=block_sizes,
        )

        self.req_output_token_ids: list[Optional[list[int]]] = []

        # Sampling-related.
        self.sampling = SamplingInputBatch(max_num_reqs)

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        """True iff all active requests are greedy (temperature == 0.0)."""
        return len(self.random_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        """True iff no active request has sampling penalties."""
        return (len(self.presence_penalties_reqs) == 0
                and len(self.frequency_penalties_reqs) == 0
                and len(self.repetition_penalties_reqs) == 0)

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: Optional[int] = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs, (
            f"req_index={req_index} >= max_num_reqs={self.max_num_reqs}")

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.token_ids_cpu[
            req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index,
                           start_idx:end_idx] = request.output_token_ids
        # Number of token ids in token_ids_cpu.
        self.num_tokens[req_index] = request.num_tokens

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)

        # Sampling-related.
        sampling_params = request.sampling_params
        assert sampling_params is not None, "pooling requests not supported yet"

        # Register with batch update builder for logits processors
        self.sampling.batch_update_builder.added.append(
            (req_index, sampling_params, request.output_token_ids))

        self.sampling.temperature[req_index] = sampling_params.temperature
        self.sampling.top_p[req_index] = sampling_params.top_p
        top_k = sampling_params.top_k
        if not (0 < top_k < self.vocab_size):
            # Normalize top_k <= 0 or >= vocab_size to vocab_size
            # (consider all tokens)
            top_k = self.vocab_size
        self.sampling.top_k[req_index] = top_k
        self.sampling.presence_penalty[req_index] = (
            sampling_params.presence_penalty)
        self.sampling.frequency_penalty[req_index] = (
            sampling_params.frequency_penalty)
        self.sampling.repetition_penalty[req_index] = (
            sampling_params.repetition_penalty)
        # Store seed, using sentinel value for None
        self.sampling.seed[req_index] = (sampling_params.seed
                                         if sampling_params.seed is not None
                                         else SEED_NONE_SENTINEL)

        # Update fast-path bookkeeping sets.
        # NOTE: Use `discard()` because `req_id` can be reused (abort+resubmit)
        # and slots can be overwritten.
        if sampling_params.temperature == 0.0:
            self.random_reqs.discard(req_id)
        else:
            self.random_reqs.add(req_id)
        if sampling_params.presence_penalty == 0.0:
            self.presence_penalties_reqs.discard(req_id)
        else:
            self.presence_penalties_reqs.add(req_id)
        if sampling_params.frequency_penalty == 0.0:
            self.frequency_penalties_reqs.discard(req_id)
        else:
            self.frequency_penalties_reqs.add(req_id)
        if sampling_params.repetition_penalty == 1.0:
            self.repetition_penalties_reqs.discard(req_id)
        else:
            self.repetition_penalties_reqs.add(req_id)

        # Generator for random sampling
        if request.generator is not None:
            self.sampling.generators[req_index] = request.generator

        # Logprobs
        if sampling_params.logprobs is not None:
            self.sampling.num_logprobs[req_index] = sampling_params.logprobs
        else:
            self.sampling.num_logprobs[req_index] = 0

        # Allowed token IDs
        if sampling_params.allowed_token_ids:
            self.sampling.has_allowed_token_ids.add(req_id)
            if self.sampling.allowed_token_ids_mask is None:
                # Lazy allocation for this tensor, which can be large.
                # True means we fill with -inf (disallowed).
                self.sampling.allowed_token_ids_mask = torch.zeros(
                    self.max_num_reqs,
                    self.vocab_size,
                    dtype=torch.bool,
                    device="cpu")
            self.sampling.allowed_token_ids_mask[req_index] = True
            # False means we don't fill with -inf (allowed).
            self.sampling.allowed_token_ids_mask[req_index][
                sampling_params.allowed_token_ids] = False

        # Bad words
        if sampling_params.bad_words_token_ids:
            self.sampling.bad_words_token_ids[
                req_index] = sampling_params.bad_words_token_ids

    def remove_request(self, req_id: str) -> Optional[int]:
        """This method must always be followed by a call to condense()."""

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self.sampling.batch_update_builder.removed_append(req_index)
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        # Update fast-path bookkeeping sets.
        self.random_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)

        # Clean up host-only sampling param tracking
        self.sampling.generators.pop(req_index, None)
        self.sampling.has_allowed_token_ids.discard(req_id)
        self.sampling.bad_words_token_ids.pop(req_index, None)

        return req_index

    def condense(self, empty_req_indices: list[int]) -> None:
        """Move non-empty requests down into lower, empty indices.
        
        Args:
            empty_req_indices: empty batch indices, sorted descending.
        """
        num_reqs = self.num_reqs
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
            empty_index = empty_req_indices.pop()
            if empty_index >= last_req_index:
                break

            # Track the move for logits processors
            self.sampling.batch_update_builder.moved.append(
                (last_req_index, empty_index,
                 MoveDirectionality.UNIDIRECTIONAL))

            # Swap the states.
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
            self.num_tokens[empty_index] = num_tokens
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[
                last_req_index]
            self.num_computed_tokens_cpu[
                empty_index] = self.num_computed_tokens_cpu[last_req_index]
            self.block_table.move_row(last_req_index, empty_index)

            # Sampling-related.
            sampling = self.sampling
            sampling.temperature[empty_index] = sampling.temperature[
                last_req_index]
            sampling.top_p[empty_index] = sampling.top_p[last_req_index]
            sampling.top_k[empty_index] = sampling.top_k[last_req_index]
            sampling.presence_penalty[empty_index] = (
                sampling.presence_penalty[last_req_index])
            sampling.frequency_penalty[empty_index] = (
                sampling.frequency_penalty[last_req_index])
            sampling.repetition_penalty[empty_index] = (
                sampling.repetition_penalty[last_req_index])
            sampling.seed[empty_index] = sampling.seed[last_req_index]
            sampling.num_logprobs[empty_index] = sampling.num_logprobs[
                last_req_index]

            # Move host-only sampling params
            if last_req_index in self.sampling.generators:
                self.sampling.generators[
                    empty_index] = self.sampling.generators.pop(last_req_index)

            if last_req_index in self.sampling.bad_words_token_ids:
                self.sampling.bad_words_token_ids[
                    empty_index] = self.sampling.bad_words_token_ids.pop(
                        last_req_index)

            # Move allowed_token_ids_mask row
            if self.sampling.allowed_token_ids_mask is not None:
                self.sampling.allowed_token_ids_mask[
                    empty_index] = self.sampling.allowed_token_ids_mask[
                        last_req_index]

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # Trim lists to the batch size.
        del self._req_ids[self.num_reqs:]
        del self.req_output_token_ids[self.num_reqs:]

    @property
    def max_num_logprobs(self) -> int:
        """Returns the maximum logprobs value across all requests, or None."""
        if self.num_reqs == 0:
            return 0
        max_val = int(self.sampling.num_logprobs[:self.num_reqs].max().item())
        return max_val

    @property
    def no_allowed_token_ids(self) -> bool:
        """True if no requests have allowed_token_ids set."""
        return len(self.sampling.has_allowed_token_ids) == 0

    def refresh_logitsprocs(self) -> None:
        """Update logits processors with batch state changes."""
        batch_update = self.sampling.batch_update_builder.get_and_reset(
            self.num_reqs)
        if batch_update:
            for processor in self.sampling.logitsprocs.all:
                processor.update_state(batch_update)

    def make_prompt_token_ids_tensor(self) -> torch.Tensor:
        """Create a tensor of prompt token IDs, padded with -1.

        NOTE: TT device sampling relies on -1 as the padding sentinel.
        If these tokens are passed to the host sampler for penalties, they must
        be canonicalized (cast to int64 and -1 replaced with vocab_size) before
        scatter operations.
        """
        max_prompt_len = (self.num_prompt_tokens[:self.num_reqs].max()
                          if self.num_reqs > 0 else 0)
        prompt_token_ids_tensor = torch.full(
            (self.num_reqs, max_prompt_len),
            -1,
            device="cpu",
            dtype=torch.int32,
        )
        prompt_token_ids = prompt_token_ids_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[:self.
                                                 num_reqs, :max_prompt_len]
        # Pad with -1 for positions beyond actual prompt length
        for i in range(self.num_reqs):
            prompt_token_ids[i, self.num_prompt_tokens[i]:] = -1
        return prompt_token_ids_tensor

    def make_output_token_ids_tensor(self) -> torch.Tensor:
        """Create a tensor of output token IDs, padded with -1.

        NOTE: TT device sampling relies on -1 as the padding sentinel.
        If these tokens are used by the host sampler penalties logic, -1 padding
        should be removed/handled before use.
        """
        output_lens = (self.num_tokens[:self.num_reqs] -
                       self.num_prompt_tokens[:self.num_reqs])
        max_output_len = int(output_lens.max()) if self.num_reqs > 0 else 0

        output_token_ids_tensor = torch.full(
            (self.num_reqs, max_output_len),
            -1,
            device="cpu",
            dtype=torch.int32,
        )
        output_token_ids = output_token_ids_tensor.numpy()
        # Copy output tokens from token_ids_cpu
        for i in range(self.num_reqs):
            prompt_len = self.num_prompt_tokens[i]
            total_len = self.num_tokens[i]
            output_len = total_len - prompt_len
            if output_len > 0:
                output_token_ids[i, :output_len] = self.token_ids_cpu[
                    i, prompt_len:total_len]
        return output_token_ids_tensor

    def advance_generators(self) -> None:
        # This relies on the fact, that for a torch all_gather_object,
        # the local object is also copied,
        # so the original object is not modified.
        # Otherwise, the generator at local_rank 0
        # would get out of sync with the others.
        for generator in self.sampling.generators.values():
            # Sample once from the generator to advance its state.
            torch.rand(1, generator=generator)
