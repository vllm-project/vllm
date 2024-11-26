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

    def _top_logprobs_token_indices(
        self,
        logprobs: torch.Tensor,
        max_num_logprobs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute top logprobs and associated token indices
        
        Args:
          logprobs: total_tokens x vocab tensor
          max_num_logprobs: Max number of top {sample,prompt} logprobs
                            requested in batch (depending on whether top sample
                            logprobs or top prompt logprobs are being computed)

        Returns:
          Top logprobs, total_tokens x max_num_logprobs tensor
          Top logprob token indices, total_tokens x max_num_logprobs tensor
        """
        topk_logprobs, topk_indices = torch.topk(logprobs,
                                                 max_num_logprobs,
                                                 dim=-1)
        # Use int32 to reduce the tensor size.
        return topk_logprobs, topk_indices.to(torch.int32)

    def _compute_logprobs_from_processed_logits(
        self,
        do_logprobs: bool,
        do_prompt_logprobs: bool,
        maybe_sampled: torch.Tensor,
        maybe_sample_logits_indices: Optional[torch.Tensor],
        prompt_logits_mask: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
        maybe_sample_logits_w_tmp_tpk_tpp: torch.Tensor,
        logits_w_tmp_tpk_tpp: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute sample and prompt logprobs as required by batch config
        
        Consumes logits which have already had temperature, top-k and top-p
        applied. 
         
        `do_logprobs` and `do_prompt_logprobs` control whether sample and
        prompt logprobs are computed, respectively.

        This function does not handle the case where no logprobs are required
        at the batch level; it is assumed this function will not be called in
        that scenario.

        Args:
          do_logprobs: compute sample logprobs
          do_prompt_logprobs: compute prompt logprobs
          maybe_sampled: list of sampled tokens; if there is a partial request,
                         includes the partial request's sampled token (which
                         will later be discarded.)
          maybe_sample_logits_indices: sequence-offset indices where a new
                         token is decoded; if there is a partial request,
                         includes the index of the partial request's sampled
                         token (which will later be discarded.)
          prompt_logits_mask: mask indicating the sequence offsets of prompt
                         tokens. Note: if there is a partial request,
                         this mask includes the index of the partial request's
                         sample token (since this sampled token will be
                         discarded, but the logprobs computed at this offset
                         are part of the prompt logprobs.) Note that this means
                         prompt_logits_mask and maybe_sample_logits_indices
                         may have overlap.
          sampling_metadata
          maybe_sample_logits_w_tmp_tpk_tpp: assumed to be logits gathered
                         from sequence offsets where a new token is being
                         decoded (including for a partial request); assumed
                         that temperature, top-k and top-p have been applied.
          logits_w_tmp_tpk_tpp: optional; all logits with temperature, top-k,
                         top-p applied.

          Returns:
            Sample logprobs (`None` if `do_logprobs == False`,
                             o/w num_samples x max_num_logprobs tensor)
            Sample logprobs token indices (`None` if `do_logprobs == False`,
                             o/w num_samples x max_num_logprobs tensor)
            Prompt logprobs (`None` if `do_prompt_logprobs == False`,
                             o/w num_prompt_tokens x max_num_prompt_logprobs
                             tensor)
            Prompt logprobs token indices (`None` if
                 `do_prompt_logprobs == False`, o/w
                 num_prompt_tokens x max_num_prompt_logprobs tensor)
        """

        assert do_logprobs or do_prompt_logprobs
        if do_logprobs and do_prompt_logprobs:
            # Batch requires sample and prompt logprobs

            # - Compute logprobs for all sequence offsets
            logprobs = self.get_logprobs(logits_w_tmp_tpk_tpp)

            # - Compute *top* logprobs for sequence offsets
            #   where a new token is being decoded
            (
                maybe_sample_topk_logprobs,
                maybe_sample_topk_indices,
            ) = self._top_logprobs_token_indices(
                logprobs[maybe_sample_logits_indices, :],
                sampling_metadata.max_num_logprobs)

            # - In case sampled tokens are not in the top logprobs at their
            #   respective sequence offsets, gather logprobs associated with
            #   sampled tokens
            maybe_sampled_logprobs = logprobs[maybe_sample_logits_indices,
                                              maybe_sampled]

            return ((
                # Sample logprobs (including sampled tokens)
                torch.cat((maybe_sample_topk_logprobs,
                           maybe_sampled_logprobs.unsqueeze(-1)),
                          dim=-1),
                # Sample logprobs token indices (including sampled tokens)
                torch.cat(
                    (maybe_sample_topk_indices, maybe_sampled.unsqueeze(-1)),
                    dim=-1)) +
                    # Prompt logprobs and token indices
                    self._top_logprobs_token_indices(
                        logprobs[prompt_logits_mask, :],
                        sampling_metadata.max_num_prompt_logprobs))
        elif do_logprobs:
            # Batch requires only sample logprobs

            # - Compute top logprobs only at sequence offsets where new tokens
            #   are being decoded
            logprobs = self.get_logprobs(maybe_sample_logits_w_tmp_tpk_tpp)
            (
                maybe_sample_topk_logprobs,
                maybe_sample_topk_indices,
            ) = self._top_logprobs_token_indices(
                logprobs, sampling_metadata.max_num_logprobs)

            # - In case sampled tokens are not in the top logprobs at their
            #   respective sequence offsets, gather logprobs associated with
            #   sampled tokens
            maybe_sampled_logprobs = logprobs[
                torch.arange(maybe_sampled.shape[0]), maybe_sampled]

            # - Concat sampled token logprobs
            maybe_sample_topk_logprobs = torch.cat(
                (maybe_sample_topk_logprobs,
                 maybe_sampled_logprobs.unsqueeze(-1)),
                dim=-1)
            # - Concat sampled token id
            maybe_sample_topk_indices = torch.cat(
                (maybe_sample_topk_indices, maybe_sampled.unsqueeze(-1)),
                dim=-1)

            # Return sample logprobs
            return (maybe_sample_topk_logprobs, maybe_sample_topk_indices,
                    None, None)

        elif do_prompt_logprobs:
            # Batch requires only prompt logprobs

            # - Compute top logprobs only at sequence offsets of prompt tokens
            logprobs = self.get_logprobs(
                logits_w_tmp_tpk_tpp[prompt_logits_mask, :])

            # Return prompt logprobs
            return ((None, None) + self._top_logprobs_token_indices(
                logprobs, sampling_metadata.max_num_prompt_logprobs))

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """Implement sampling.
        
        Apply temperature, top-k and top-p.
        Sample from the probability distribution implied by `logits`.
        Only sample at sequence offsets where new tokens are decoded.
        In the process, compute sample and prompt logprobs (if required.)

        Args:
          logits: model output logits which imply probability distribution.
          sampling_metadata: sampling config settings
        
        Returns:
          Sampler output. Sampled tokens and sample/prompt logprobs
          (if requested)
        """

        # Batch-level logprobs configs. `do_logprobs` indicates whether
        # any request requires sample logprobs. `do_prompt_logprobs`
        # indicates whether any request requires prompt logprobs.
        do_logprobs = sampling_metadata.max_num_logprobs > 0
        do_prompt_logprobs = sampling_metadata.max_num_prompt_logprobs > 0
        do_any_logprobs = do_logprobs or do_prompt_logprobs

        num_query_tokens = sampling_metadata.num_query_tokens
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        maybe_sample_logits_indices = sampling_metadata.query_start_loc[1:] - 1
        prompt_logits_mask = torch.ones(sampling_metadata.num_input_tokens,
                                        dtype=torch.bool)
        # Sequence offsets where a token is being decoded are *not* prompt
        # tokens...
        pdx = sampling_metadata.partial_req_index
        prompt_logits_mask[maybe_sample_logits_indices] = False
        # ...unless the request in question is partial
        prompt_logits_mask[maybe_sample_logits_indices[pdx]] = True

        # Apply temperature, top-k and top-p to logits at sequence offsets
        # where a new token is being decoded.
        if do_prompt_logprobs:
            # If prompt logprobs are required, then temp/top-k/top-p
            # must also be applied to prompt logits as a prerequisite.
            # So pass *all* logits through temp/top-k/top-p, then gather
            # the processed logits from the sequence offsets where a new token
            # is being decoded.
            logits_w_tmp_tpk_tpp = self._apply_temperature_top_k_top_p(
                logits, sampling_metadata, num_query_tokens)

            maybe_sample_logits_w_tmp_tpk_tpp = (
                logits_w_tmp_tpk_tpp[maybe_sample_logits_indices])
        else:
            # If prompt logprobs are not required, then gather the logits
            # only from the sequence offsets where a new token is being
            # decoded, and *only* apply temp/top-k/top-p to those logits.
            maybe_sample_logits_w_tmp_tpk_tpp = (
                self._apply_temperature_top_k_top_p(
                    logits[maybe_sample_logits_indices], sampling_metadata,
                    None))

        # Compute and sample token probability distribution, *only* at sequence
        # offsets where a new token is being decoded
        maybe_sampled = self._probs_sample(maybe_sample_logits_w_tmp_tpk_tpp,
                                           sampling_metadata)

        # Compute sample & prompt logprobs, as-needed
        if do_any_logprobs:
            (
                maybe_sample_logprobs,
                maybe_sample_logprobs_token_indices,
                prompt_logprobs,
                prompt_logprobs_token_indices,
            ) = self._compute_logprobs_from_processed_logits(
                do_logprobs=do_logprobs,
                do_prompt_logprobs=do_prompt_logprobs,
                maybe_sampled=maybe_sampled,
                maybe_sample_logits_indices=maybe_sample_logits_indices,
                prompt_logits_mask=prompt_logits_mask,
                sampling_metadata=sampling_metadata,
                maybe_sample_logits_w_tmp_tpk_tpp=
                maybe_sample_logits_w_tmp_tpk_tpp,
                logits_w_tmp_tpk_tpp=(logits_w_tmp_tpk_tpp
                                      if do_prompt_logprobs else None))

            # Return decoded output tokens and sample/prompt logprobs,
            # as required
            return SamplerOutput(
                sampled_token_ids=maybe_sampled,
                logprobs=maybe_sample_logprobs,
                logprob_token_ids=maybe_sample_logprobs_token_indices,
                prompt_logprobs=prompt_logprobs,
                prompt_logprob_token_ids=prompt_logprobs_token_indices)
        else:
            # No logprobs; return decoded output tokens
            return SamplerOutput(sampled_token_ids=maybe_sampled)

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
