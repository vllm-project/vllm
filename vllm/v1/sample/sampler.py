# SPDX-License-Identifier: Apache-2.0
"""A layer that samples the next tokens from the model's outputs."""

import torch
import torch.nn as nn

from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.penalties import (apply_all_penalties,
                                          apply_min_token_penalties)
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.rejection_sampler import RejectionSampler

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()
        self.rejection_sampler = RejectionSampler()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        if sampling_metadata.spec_token_ids:
            if sampling_metadata.max_num_logprobs:
                raise NotImplementedError(
                    "Rejection sampling does not support logprobs.")
            return self.rejection_sampler(
                logits,
                sampling_metadata,
            )

        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        # TODO(rob): provide option for logprobs post sampling.
        # See https://vllm-dev.slack.com/archives/C07UUL8E61Z/p1735907856007919 # noqa: E501
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            raw_logprobs = self.compute_logprobs(logits)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Apply logits bias.
        logits = self.apply_logits_bias(logits, sampling_metadata)
        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata)
        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)

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
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)

        # Apply min_p.
        if sampling_metadata.min_p is not None:
            logits = self.apply_min_p(logits, sampling_metadata.min_p)

        # Apply top_k and/or top_p.
        random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
        )
        return sampled

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
          logits: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs,
                                                 num_logprobs,
                                                 dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = (logprobs >= token_logprobs).sum(-1)

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
        if sampling_metadata.min_tokens:
            apply_min_token_penalties(logits,
                                      sampling_metadata.output_token_ids,
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

    def apply_min_p(
        self,
        logits: torch.Tensor,
        min_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Filters logits using adaptive probability thresholding.
        """
        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values,
                                       dim=-1,
                                       keepdim=True)
        # Reshape min_p for broadcasting
        adjusted_min_p = min_p.unsqueeze(1) * max_probabilities
        # Identify valid tokens using threshold comparison
        valid_token_mask = probability_values >= adjusted_min_p
        # Apply mask using boolean indexing
        logits[~valid_token_mask] = -float('inf')
        return logits

    def apply_logits_bias(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        # TODO(houseroad): this implementation is extremely inefficient.
        # One idea is implement this as a PyTorch C++ op, and we may
        # even optimize the logit_bias layout.
        for i, logit_bias in enumerate(sampling_metadata.logit_bias):
            if logit_bias:
                for token_id, bias in logit_bias.items():
                    logits[i, token_id] += bias
        return logits
