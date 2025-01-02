"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import LogitsProcessMetadata, SamplingMetadata

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        logits = self._process_logits(
            logits, sampling_metadata.logits_process_metadata)
        probs = self.get_probs(logits)
        sampled = self.sample(probs, sampling_metadata)
        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # Compute the logprobs if requested.
        # NOTE: CPU-GPU synchronization happens here.
        logprob_token_ids, logprobs = self._compute_logprobs(
            logits=logits,
            max_num_logprobs=sampling_metadata.max_num_logprobs,
            sampled_token_ids=sampled)

        # NOTE: CPU-GPU synchronization happens here.
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled.tolist(),
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
        )
        return sampler_output

    def compute_prompt_logprobs(
        self,
        logits: torch.Tensor,
        logits_process_metadata: LogitsProcessMetadata,
        num_logprobs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self._process_logits(logits, logits_process_metadata)

        # NOTE: CPU-GPU synchronization happens here.
        logprob_token_ids, logprobs = self._compute_logprobs(
            logits=logits, max_num_logprobs=num_logprobs)

        return logprob_token_ids, logprobs

    def _compute_logprobs(
        self,
        logits: torch.Tensor,
        max_num_logprobs: int,
        sampled_token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if max_num_logprobs > 0:
            logprobs = self.get_logprobs(logits)
            topk_logprobs, topk_indices = torch.topk(logprobs,
                                                     max_num_logprobs,
                                                     dim=-1,
                                                     sorted=True)
            # Use int32 to reduce the tensor size.
            topk_indices = topk_indices.to(torch.int32)

            # Concatenate with the sampled token_id if provided.
            if sampled_token_ids:
                # TODO(rob): check if the indexing / concatting is right
                # TODO(rob): do we need to return the rank of the sampled?
                sampled_logprobs = logprobs[:, sampled_token_ids]
                topk_indices = torch.cat([sampled_token_ids, topk_indices])
                topk_logprobs = torch.cat([sampled_logprobs, topk_logprobs])

            return topk_indices.cpu(), topk_logprobs.cpu()
        else:
            return None, None

    def _process_logits(
        self,
        logits: torch.Tensor,
        logits_process_metadata: LogitsProcessMetadata,
    ) -> torch.Tensor:
        logits = self._apply_temperature(logits,
                                         logits_process_metadata.temperature)
        logits = self._apply_top_k_top_p(logits, logits_process_metadata)
        return logits

    def _apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Use float32 to apply temperature scaling.
        logits = logits.to(torch.float32)
        # Avoid division by zero.
        temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        # Use in-place division to avoid creating a new tensor.
        logits.div_(temp.unsqueeze(dim=1))
        return logits

    def _apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        logits_process_metadata: LogitsProcessMetadata,
    ) -> torch.Tensor:
        return _apply_top_k_top_p(
            logits,
            logits_process_metadata.no_top_k,
            logits_process_metadata.top_k,
            logits_process_metadata.no_top_p,
            logits_process_metadata.top_p,
        )

    def get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1, dtype=torch.float32)

    def get_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(logits, dim=-1, dtype=torch.float32)

    def greedy_sample(self, probs: torch.Tensor) -> torch.Tensor:
        return probs.argmax(dim=-1).view(-1)

    def random_sample(
        self,
        probs: torch.Tensor,
        generators: Dict[int, torch.Generator],
    ) -> torch.Tensor:
        q = torch.empty_like(probs)
        # NOTE(woosuk): To batch-process the requests without their own seeds,
        # which is the common case, we first assume that every request does
        # not have its own seed. Then, we overwrite the values for the requests
        # that have their own seeds.
        if len(generators) != probs.shape[0]:
            # This might still be done here unnecessarily if there are greedies
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
        return probs.div_(q).argmax(dim=-1).view(-1)

    def sample(
        self,
        probs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_greedy:
            return self.greedy_sample(probs)
        if sampling_metadata.all_random:
            return self.random_sample(probs, sampling_metadata.generators)

        greedy_sampled = self.greedy_sample(probs)
        random_sampled = self.random_sample(probs,
                                            sampling_metadata.generators)
        temperature = sampling_metadata.logits_process_metadata.temperature
        sampled = torch.where(temperature < _SAMPLING_EPS, greedy_sampled,
                              random_sampled)
        return sampled


# TODO(woosuk): Optimize this with a custom kernel.
def _apply_top_k_top_p(
    logits: torch.Tensor,
    no_top_k: bool,
    k: torch.Tensor,
    no_top_p: bool,
    p: torch.Tensor,
) -> torch.Tensor:
    if no_top_k and no_top_p:
        return logits
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if not no_top_k:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if not no_top_p:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits
