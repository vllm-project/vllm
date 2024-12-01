"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from vllm.utils import apply_sampling_penalties, make_tensor_with_pad
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        logits = self.apply_top_k_top_p(logits, sampling_metadata)
        _apply_min_token_penalties(logits, sampling_metadata.output_token_ids,
                                   sampling_metadata.stop_token_ids,
                                   sampling_metadata.min_tokens)
        _apply_penalties(logits, sampling_metadata.prompt_token_ids,
                         sampling_metadata.output_token_ids,
                         sampling_metadata.presence_penalties,
                         sampling_metadata.frequency_penalties,
                         sampling_metadata.repetition_penalties)
        probs = self.get_probs(logits)
        sampled = self.sample(probs, sampling_metadata)
        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        if sampling_metadata.max_num_logprobs > 0:
            logprobs = self.get_logprobs(logits)
            # FIXME: Mask the sampled token_id, get topk logprobs,
            # and concatenate the topk with the sampled token_id.
            topk_logprobs, topk_indices = torch.topk(
                logprobs, sampling_metadata.max_num_logprobs, dim=-1)
            # Use int32 to reduce the tensor size.
            topk_indices = topk_indices.to(torch.int32)
        else:
            topk_logprobs = None
            topk_indices = None

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
        # Use float32 to apply temperature scaling.
        logits = logits.to(torch.float32)
        # Avoid division by zero.
        temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        # Use in-place division to avoid creating a new tensor.
        logits.div_(temp.unsqueeze(dim=1))
        return logits

    def apply_top_k_top_p(
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


def _apply_min_token_penalties(logits: torch.Tensor,
                               output_token_ids: List[List[int]],
                               stop_token_ids: List[List[int]],
                               min_tokens: List[int]):
    # Compute min_tokens_logits_to_penalize
    min_tokens_logits_to_penalize: List[Tuple[int, int]] = []
    for index, min_token in enumerate(min_tokens):
        if (min_token > 0 and len(output_token_ids[index]) < min_token):
            for stop_token_id in stop_token_ids:
                min_tokens_logits_to_penalize.append((index, stop_token_id))
    if min_tokens_logits_to_penalize:
        logits[tuple(zip(*min_tokens_logits_to_penalize))] = -float("inf")


def _apply_penalties(logits: torch.Tensor, prompt_token_ids: List[List[int]],
                     output_token_ids: List[List[int]],
                     presence_penalties: List[float],
                     frequency_penalties: List[float],
                     repetition_penalties: List[float]):
    apply_penalties = any(p != 0.0 for p in presence_penalties) or any(
        f != 0.0
        for f in frequency_penalties) or any(r != 1.0
                                             for r in repetition_penalties)
    if apply_penalties:
        # Convert to tensors
        _, vocab_size = logits.shape
        (prompt_tokens_t, output_tokens_t, frequency_penalties_t,
        presence_penalties_t, repetition_penalties_t) = \
            _convert_to_tensors(
                prompt_token_ids, output_token_ids, frequency_penalties,
                presence_penalties, repetition_penalties, vocab_size,
                logits.device)
        return apply_sampling_penalties(logits, prompt_tokens_t,
                                        output_tokens_t, presence_penalties_t,
                                        frequency_penalties_t,
                                        repetition_penalties_t)


def _convert_to_tensors(prompt_token_ids: List[List[int]],
                        output_token_ids: List[List[int]],
                        frequency_penalties: List[float],
                        presence_penalties: List[float],
                        repetition_penalties: List[float], vocab_size: int,
                        device: torch.device) -> Tuple[torch.Tensor, ...]:
    prompt_tokens_tensor = make_tensor_with_pad(
        prompt_token_ids,
        vocab_size,
        device=device,
        dtype=torch.int64,
    )
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        vocab_size,
        device=device,
        dtype=torch.int64,
    )
    frequency_penalties_tensor = torch.tensor(
        frequency_penalties,
        device=device,
        dtype=torch.float,
    )
    presence_penalties_tensor = torch.tensor(
        presence_penalties,
        device=device,
        dtype=torch.float,
    )
    repetition_penalties_tensor = torch.tensor(
        repetition_penalties,
        device=device,
        dtype=torch.float,
    )

    return (prompt_tokens_tensor, output_tokens_tensor,
            frequency_penalties_tensor, presence_penalties_tensor,
            repetition_penalties_tensor)
