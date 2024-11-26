"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):

    def _apply_temperature_top_k_top_p(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        num_query_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:

        temperature = (sampling_metadata.temperature if
                       num_query_tokens is None else torch.repeat_interleave(
                           sampling_metadata.temperature, num_query_tokens))

        return self._apply_top_k_top_p(
            self._apply_temperature(logits, temperature), sampling_metadata)

    def _probs_sample(
        self,
        maybe_sample_logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        probs = self.get_probs(maybe_sample_logits)
        sampled = self.sample(probs, sampling_metadata)
        # Use int32 to reduce the tensor size.
        return sampled.to(torch.int32)

    def _topk_logprobs_indices(
        self,
        logprobs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        topk_logprobs, topk_indices = torch.topk(
            logprobs, sampling_metadata.max_num_logprobs, dim=-1)
        # Use int32 to reduce the tensor size.
        return topk_logprobs, topk_indices.to(torch.int32)

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:

        do_logprobs = sampling_metadata.max_num_logprobs > 0
        do_prompt_logprobs = sampling_metadata.max_num_prompt_logprobs > 0
        num_query_tokens = sampling_metadata.num_query_tokens
        maybe_sample_logits_indices = (
            sampling_metadata.maybe_sample_logits_indices)
        prompt_logits_mask = sampling_metadata.prompt_logits_mask

        if do_prompt_logprobs:
            logits_w_tmp_tpk_tpp = self._apply_temperature_top_k_top_p(
                logits, sampling_metadata, num_query_tokens)

            maybe_sample_logits_w_tmp_tpk_tpp = (
                logits_w_tmp_tpk_tpp[maybe_sample_logits_indices])
        else:
            maybe_sample_logits_w_tmp_tpk_tpp = (
                self._apply_temperature_top_k_top_p(
                    logits[maybe_sample_logits_indices], sampling_metadata,
                    None))

        maybe_sampled = self._probs_sample(maybe_sample_logits_w_tmp_tpk_tpp,
                                           sampling_metadata)

        if do_logprobs and do_prompt_logprobs:
            logprobs = self.get_logprobs(logits_w_tmp_tpk_tpp)

            maybe_sampled_logprobs = logprobs[maybe_sample_logits_indices,
                                              maybe_sampled]

            topk_logprobs, topk_indices = self._topk_logprobs_indices(
                logprobs, sampling_metadata)

            maybe_sample_topk_logprobs = topk_logprobs[
                maybe_sample_logits_indices, :]
            maybe_sample_topk_indices = topk_indices[
                maybe_sample_logits_indices, :]
            prompt_topk_logprobs = topk_logprobs[prompt_logits_mask, :]
            prompt_topk_indices = topk_indices[prompt_logits_mask, :]

            # Concat sampled token logprobs
            maybe_sample_topk_logprobs = torch.cat(
                (maybe_sample_topk_logprobs,
                 maybe_sampled_logprobs.unsqueeze(-1)),
                dim=-1)
            #Concat sampled token id
            maybe_sample_topk_indices = torch.cat(
                (maybe_sample_topk_indices, maybe_sampled.unsqueeze(-1)),
                dim=-1)
        elif do_logprobs:
            logprobs = self.get_logprobs(
                logits_w_tmp_tpk_tpp[maybe_sample_logits_indices, :])

            maybe_sampled_logprobs = logprobs[
                torch.arange(maybe_sampled.shape[0]), maybe_sampled]

            (
                maybe_sample_topk_logprobs,
                maybe_sample_topk_indices,
            ) = self._topk_logprobs_indices(logprobs, sampling_metadata)

            # Concat sampled token logprobs
            maybe_sample_topk_logprobs = torch.cat(
                (maybe_sample_topk_logprobs,
                 maybe_sampled_logprobs.unsqueeze(-1)),
                dim=-1)
            #Concat sampled token id
            maybe_sample_topk_indices = torch.cat(
                (maybe_sample_topk_indices, maybe_sampled.unsqueeze(-1)),
                dim=-1)

            (
                prompt_topk_logprobs,
                prompt_topk_indices,
            ) = (None, None)

        elif do_prompt_logprobs:
            logprobs = self.get_logprobs(
                logits_w_tmp_tpk_tpp[prompt_logits_mask, :])

            prompt_topk_logprobs, prompt_topk_indices = (
                self._topk_logprobs_indices(logprobs, sampling_metadata))

            (
                maybe_sample_topk_logprobs,
                maybe_sample_topk_indices,
            ) = (None, None)
        else:
            (
                maybe_sample_topk_logprobs,
                maybe_sample_topk_indices,
                prompt_topk_logprobs,
                prompt_topk_indices,
            ) = (None, None, None, None)

        sampler_output = SamplerOutput(
            sampled_token_ids=maybe_sampled,
            logprob_token_ids=maybe_sample_topk_indices,
            logprobs=maybe_sample_topk_logprobs,
            prompt_logprob_token_ids=prompt_topk_indices,
            prompt_logprobs=prompt_topk_logprobs)

        return sampler_output

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
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return _apply_top_k_top_p(
            logits,
            sampling_metadata.no_top_k,
            sampling_metadata.top_k,
            sampling_metadata.no_top_p,
            sampling_metadata.top_p,
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
        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
        )
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
