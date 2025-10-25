# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampler layer implementing TPU supported operations."""

import torch
import torch.nn as nn

from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.tpu.metadata import TPUSupportedSamplingMetadata

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):
    def __init__(self):
        # TODO(houseroad): Add support for logprobs_mode.
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: TPUSupportedSamplingMetadata,
    ) -> SamplerOutput:
        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)

        # These are TPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=None,
        )
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool = False,
    ) -> torch.Tensor:
        # Avoid division by zero for greedy sampling (temperature ~ 0.0).
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: TPUSupportedSamplingMetadata,
    ) -> torch.Tensor:
        greedy_sampled = self.greedy_sample(logits)

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        # Apply min_p.
        if sampling_metadata.min_p is not None:
            logits = self.apply_min_p(logits, sampling_metadata.min_p)

        # Apply top_k and/or top_p.
        logits = apply_top_k_top_p(
            logits,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        # Random sample.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        random_sampled = self.random_sample(probs, sampling_metadata.generators)

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
          logprobs: (num tokens) x (vocab) tensor
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
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

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
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        # Reshape min_p for broadcasting
        adjusted_min_p = min_p.unsqueeze(1) * max_probabilities
        # Identify valid tokens using threshold comparison
        valid_token_mask = probability_values >= adjusted_min_p
        # Apply mask using boolean indexing (xla friendly)
        logits.masked_fill_(~valid_token_mask, -float("inf"))
        return logits

    def random_sample(
        self,
        probs: torch.Tensor,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor:
        q = torch.empty_like(probs)
        # NOTE(woosuk): To batch-process the requests without their own seeds,
        # which is the common case, we first assume that every request does
        # not have its own seed. Then, we overwrite the values for the requests
        # that have their own seeds.
        q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
        return probs.div_(q).argmax(dim=-1).view(-1)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """
    Apply top-k and top-p optimized for TPU.

    This algorithm avoids using torch.scatter which is extremely slow on TPU.
    This is achieved by finding a "cut-off" element in the original logit, and
    after thresholding the logit using this cut-off, the remaining elements
    shall constitute the top-p set.

    Note: in the case of tie (i.e. multipple cut-off elements present in the
    logit), all tie elements are included in the top-p set. In other words,
    this function does not break ties. Instead, these tie tokens have equal
    chance of being chosen during final sampling, so we can consider the tie
    being broken then.
    """
    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    return logits
