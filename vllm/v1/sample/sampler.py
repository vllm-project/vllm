"""A layer that samples the next tokens from the model's outputs."""
from typing import Tuple, Optional, Tuple

import torch
import torch.nn as nn

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import LogitsProcessMetadata, SamplingMetadata
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
            # NOTE(rob): We have to clone the raw logits (at fp16) to
            # compute logprobs AFTER sampling, since we need return
            # the logprob of the sampled token.
            raw_logits = torch.empty_like(logits)
            raw_logits.copy_(logits, non_blocking=True)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata)
        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)
        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # Compute topk and sample logprobs.
        logprob_token_ids, logprobs = self.get_logprobs(
            raw_logits,
            sampling_metadata.max_num_logprobs,
            sampled=sampled,
        )

        # NOTE: CPU-GPU synchronization happens here.
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled.tolist(),
            logprob_token_ids=logprob_token_ids or logprob_token_ids.cpu(),
            logprobs=logprobs or logprobs.cpu(),
        )
        return sampler_output

    def get_prompt_logprobs(
        self,
        logits: torch.Tensor,
        logits_process_metadata: LogitsProcessMetadata,
        num_logprobs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.apply_temperature(logits,
                                        logits_process_metadata.temperature)
        logits = self.apply_top_k_top_p(logits, logits_process_metadata)

        # NOTE: CPU-GPU synchronization happens here.
        logprob_token_ids, logprobs = self._compute_logprobs(
            logits=logits, max_num_logprobs=num_logprobs)

        return logprob_token_ids, logprobs

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
            sampling_metadata.logits_process_metadata.temperature <
            _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
        )
        return sampled

    def get_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        sampled: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if num_logprobs == 0:
            return None, None

        topk_logprobs, topk_indices = torch.topk(logprobs,
                                                 num_logprobs,
                                                 dim=-1)
        # Use int32 to reduce the tensor size.
        topk_indices = topk_indices.to(torch.int32)

        # Concatenate with the sampled token_id if provided.
        if sampled:
            # TODO(andy): do we need to return the rank of the sampled?
            # TODO(andy): is this indexing right?
            sampled_logprobs = logprobs[:, sampled]
            topk_indices = torch.cat([sampled, topk_indices])
            topk_logprobs = torch.cat([sampled_logprobs, topk_logprobs])

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
