# SPDX-License-Identifier: Apache-2.0
"""A layer that samples the next tokens from the model's outputs."""
from typing import Tuple

import torch
import torch.nn as nn

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.penalties import (apply_all_penalties,
                                          apply_min_token_penalties)
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        needs_logprobs = sampling_metadata.max_num_logprobs > 0
        if needs_logprobs:
            # NOTE(woosuk): Use the original logits (before any penalties or
            # temperature scaling) for the top-k logprobs.
            # This is different from the V0 sampler, which uses the logits that
            # is used for sampling (after penalties and temperature scaling).
            # NOTE: We compute logprobs first because the below ops may
            # modify the logits tensor in-place (and we don't want to clone
            # the logits tensor for memory efficiency).
            topk_logprobs, topk_indices = self.get_topk_logprobs(
                logits, sampling_metadata)
        else:
            topk_logprobs = None
            topk_indices = None

        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata)
        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        # Apply logits processors.
        logits = self.apply_logits_processors(logits, sampling_metadata)
        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)
        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        sampler_output = SamplerOutput(
            sampled_token_ids=sampled,
            logprob_token_ids=topk_indices,
            logprobs=topk_logprobs,
            prompt_logprob_token_ids=None,
            prompt_logprobs=None,
        )
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Avoid division by zero.
        temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        # Use in-place division to avoid creating a new tensor.
        logits.div_(temp.unsqueeze(dim=1))
        return logits

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_greedy:
            return self.greedy_sample(logits)

        random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.no_top_k,
            sampling_metadata.top_k,
            sampling_metadata.no_top_p,
            sampling_metadata.top_p,
        )
        if sampling_metadata.all_random:
            return random_sampled

        greedy_sampled = self.greedy_sample(logits)
        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
        )
        return sampled

    def get_topk_logprobs(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
        # FIXME: Mask the sampled token_id, get topk logprobs,
        # and concatenate the topk with the sampled token_id.
        topk_logprobs, topk_indices = torch.topk(
            logprobs, sampling_metadata.max_num_logprobs, dim=-1)
        # Use int32 to reduce the tensor size.
        topk_indices = topk_indices.to(torch.int32)
        return topk_logprobs, topk_indices

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        apply_min_token_penalties(logits, sampling_metadata.output_token_ids,
                                  sampling_metadata.stop_token_ids,
                                  sampling_metadata.min_tokens)
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = apply_all_penalties(
                logits, sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids)
        return logits

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if len(sampling_metadata.logits_processors) == 0:
            return logits

        for seq_idx, seq_logits_processors in \
            sampling_metadata.logits_processors.items():
            logits_row = logits[seq_idx]
            original_logits_row = logits_row
            past_tokens_ids = sampling_metadata.output_token_ids[seq_idx]
            prompt_tokens_ids = sampling_metadata.prompt_token_ids_cpu[seq_idx]

            for logits_processor in seq_logits_processors:
                logits_row = logits_processor(prompt_tokens_ids,
                                              past_tokens_ids, logits_row)

            if logits_row is not original_logits_row:
                logits[seq_idx] = logits_row

        return logits
