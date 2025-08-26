# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that samples the next tokens from the model's outputs."""

from typing import Optional

import torch
import torch.nn as nn

from vllm.config import LogprobsMode
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:

    1. If logprobs are requested:
        a) If `logprobs_mode` is `raw_logprobs`, compute logprobs
           as the final logprobs to return.
        b) If `logprobs_mode` is `raw_logits`, clone the logits
           as the final logprobs to return.
    2. Convert logits to float32.
    3. Apply allowed token ids whitelist.
    4. Apply bad words exclusion.
    5. Apply logit processors which are not argmax-invariant,
       i.e. that can impact greedy sampling.
        a) Min tokens processor
        b) Logit bias processor
    6. Apply penalties
        a) Repetition penalty
        b) Frequency penalty
        c) Presence penalty
    7. Sample the next tokens. `sample` method performs the following steps:
        a) If not `all_random`, perform greedy sampling. If `all_greedy`,
           return the greedily sampled tokens and final logprobs if requested.
        b) Apply temperature.
        c) Apply logit processors which are argmax-invariant, by default
           the min_p processor.
        d) Apply top_k and/or top_p.
        e) Sample the next tokens with the probability distribution.
        f) If `all_random` or temperature >= epsilon (1e-5), return the
           randomly sampled tokens and final logprobs if requested. Else,
           return the greedily sampled tokens and logprobs if requested.
    8. Gather the logprobs of the top `max_num_logprobs` and sampled token
       (if requested). Note that if the sampled token is within the top
       `max_num_logprobs`, the logprob will be eventually merged in
       `LogprobsProcessor` during output processing. Therefore, the
       final output may contain either `max_num_logprobs + 1` or
       `max_num_logprobs` logprobs.
    9. Return the final `SamplerOutput`.
    """

    def __init__(self,
                 logprobs_mode: LogprobsMode = LogprobsMode.RAW_LOGPROBS):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode)
        self.pin_memory = is_pin_memory_available()
        self.logprobs_mode = logprobs_mode

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if self.logprobs_mode == LogprobsMode.RAW_LOGPROBS:
                raw_logprobs = self.compute_logprobs(logits)
            elif self.logprobs_mode == LogprobsMode.RAW_LOGITS:
                raw_logprobs = logits.clone()

        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Apply allowed token ids.
        logits = self.apply_allowed_token_ids(logits, sampling_metadata)
        # Apply bad words exclusion.
        logits = self.apply_bad_words(logits, sampling_metadata)

        # Apply logits processors which can impact greedy sampling
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata)

        # Sample the next token.
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        # Gather the logprobs of the topk and sampled token (if requested).
        # Get logprobs and rank tensors (if requested)
        logprobs_tensors = None if num_logprobs is None else \
            self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample_single_rank(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample logits based on sampling metadata on a single rank.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if self.logprobs_mode == LogprobsMode.PROCESSED_LOGITS:
                        processed_logprobs = logits
                    elif self.logprobs_mode == LogprobsMode.PROCESSED_LOGPROBS:
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top_k and/or top_p.
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample logits based on sampling metadata.
        Can parallelize sampling across tensor parallel ranks.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """
        world_size = get_tensor_model_parallel_world_size()
        assert logits.ndim == 2, f"Logits should be 2D, but got {logits.shape=}"
        batch_size = logits.shape[0]

        # Skip parallelization and fall back to single-rank sampling if:
        # 1. world_size < 2: Only one TP rank, no benefit from parallelization
        # 2. batch_size < 8: Too small batch, overhead outweighs benefits
        # 3. Custom generators present: Per-request generators complicate synchronization
        #    across ranks since different ranks would need different generator states
        has_generators = sampling_metadata.generators and len(sampling_metadata.generators) > 0
        use_single_rank = (world_size < 2 or batch_size < 8 or has_generators)

        if use_single_rank:
            return self.sample_single_rank(logits, sampling_metadata)

        # Calculate chunk size for distributing batch across TP ranks
        # Use ceiling division to handle uneven splits
        chunk_size = (batch_size + world_size - 1) // world_size

        # Calculate start/end indices for each rank's chunk
        start_indices = [min(i * chunk_size, batch_size) for i in range(world_size)]
        end_indices = [min((i + 1) * chunk_size, batch_size) for i in range(world_size)]
        rank = get_tensor_model_parallel_rank()
        start_idx = start_indices[rank]
        end_idx = end_indices[rank]

        # Handle generator mapping for the local chunk
        # Since we already checked that generators dict is empty above (condition 3),
        # this will always result in an empty local_generators dict.
        # However, we keep this logic for potential future use cases where
        # we might allow parallelization with limited generator usage.
        local_generators = {}
        if batch_size % world_size == 0:
            # Even split case: safe to map generators normally since all ranks
            # get identical chunk sizes, maintaining RNG synchronization
            for key in sampling_metadata.generators:
                if start_idx <= key < end_idx:
                    # Remap global generator index to local chunk index
                    local_generators[key - start_idx] = sampling_metadata.generators[key]
        else:
            # Uneven split case: different chunk sizes across ranks would cause
            # RNG divergence, so we use empty generators to stay synchronized.
            # All random sampling will use the default RNG state.
            local_generators = {}

        # Create local sampling metadata for this rank's chunk by slicing all tensors
        # to only include the portion this rank is responsible for
        local_sampling_metadata = SamplingMetadata(
            # Slice temperature tensor to local chunk size
            temperature=sampling_metadata.temperature[start_idx:end_idx]
                       if sampling_metadata.temperature is not None else None,
            # Copy boolean flags (apply to all chunks)
            all_greedy=sampling_metadata.all_greedy,
            all_random=sampling_metadata.all_random,
            # Slice top_k tensor to local chunk size
            top_k=sampling_metadata.top_k[start_idx:end_idx]
                  if sampling_metadata.top_k is not None else None,
            # Slice top_p tensor to local chunk size
            top_p=sampling_metadata.top_p[start_idx:end_idx]
                  if sampling_metadata.top_p is not None else None,
            # Use remapped generators for local chunk
            generators=local_generators,
            # Keep global settings unchanged
            max_num_logprobs=sampling_metadata.max_num_logprobs,
            # Copy boolean flag (apply to all chunks)
            no_penalties=sampling_metadata.no_penalties,
            # Slice all penalty and token arrays to local chunk
            prompt_token_ids=sampling_metadata.prompt_token_ids[start_idx:end_idx]
                           if sampling_metadata.prompt_token_ids is not None else None,
            frequency_penalties=sampling_metadata.frequency_penalties[start_idx:end_idx]
                              if sampling_metadata.frequency_penalties is not None else None,
            presence_penalties=sampling_metadata.presence_penalties[start_idx:end_idx]
                             if sampling_metadata.presence_penalties is not None else None,
            repetition_penalties=sampling_metadata.repetition_penalties[start_idx:end_idx]
                                if sampling_metadata.repetition_penalties is not None else None,
            # Slice output token ids list to local chunk
            output_token_ids=sampling_metadata.output_token_ids[start_idx:end_idx]
                           if sampling_metadata.output_token_ids is not None else None,
            allowed_token_ids_mask=sampling_metadata.allowed_token_ids_mask[start_idx:end_idx]
                                  if sampling_metadata.allowed_token_ids_mask is not None else None,
            # Keep global settings that apply to all chunks
            bad_words_token_ids=sampling_metadata.bad_words_token_ids,
            logitsprocs=sampling_metadata.logitsprocs,
        )

        # Sample tokens locally on this rank's chunk of the batch
        tokens_local = self.sample_single_rank(
            logits[start_idx:end_idx],  # Only process this rank's logits slice
            local_sampling_metadata,    # Use metadata with local chunk parameters
        )

        # Use tensor parallel all_gather to collect results from all ranks
        is_even_chunks = batch_size % world_size == 0

        if is_even_chunks:
            # Even chunk sizes - all ranks process identical chunk sizes
            # The all_gather result is already in correct batch order, no reconstruction needed
            tokens_all = get_tp_group().all_gather(tokens_local, 0)
            return tokens_all
        else:
            # Uneven chunk sizes - fall back to single-rank sampling for now
            # TODO: Implement proper uneven chunk distributed sampling
            # This requires careful handling of different tensor sizes across ranks
            return self.sample_single_rank(logits, sampling_metadata)

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs,
                                                 num_logprobs,
                                                 dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids,
            )
        return logits

    def apply_allowed_token_ids(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask,
                                float("-inf"))
        return logits

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.bad_words_token_ids:
            apply_bad_words(
                logits,
                sampling_metadata.bad_words_token_ids,
                sampling_metadata.output_token_ids,
            )
        return logits
