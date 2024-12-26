"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn

from vllm.model_executor.layers.utils import apply_penalties
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
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
        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        _apply_min_token_penalties(logits, sampling_metadata.output_token_ids,
                                   sampling_metadata.stop_token_ids,
                                   sampling_metadata.min_tokens)
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            _apply_penalties(logits, sampling_metadata.prompt_token_ids,
                             sampling_metadata.presence_penalties,
                             sampling_metadata.frequency_penalties,
                             sampling_metadata.repetition_penalties,
                             sampling_metadata.output_token_ids)

        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        logits = self.apply_top_k_top_p(logits, sampling_metadata)
        probs = self.get_probs(logits)
        sampled = self.sample(probs, sampling_metadata)


        orig_logits = logits

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)


        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        if sampling_metadata.max_num_logprobs > 0:
            logprobs = self.get_logprobs(orig_logits)
            # FIXME: Mask the sampled token_id, get topk logprobs,
            # and concatenate the topk with the sampled token_id.
            topk_logprobs, topk_indices = torch.topk(
                logprobs, sampling_metadata.max_num_logprobs, dim=-1)
            # Use int32 to reduce the tensor size.
            topk_indices = topk_indices.to(torch.int32)
        else:
            topk_logprobs = None
            topk_indices = None

        # NOTE: CPU-GPU synchronization happens here.
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled.tolist(),
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
        return logits / temp.unsqueeze(dim=1)

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

    def get_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(logits, dim=-1, dtype=torch.float32)


def _apply_min_token_penalties(logits: torch.Tensor,
                               output_token_ids: List[List[int]],
                               stop_token_ids: List[Set[int]],
                               min_tokens: List[int]):
    """
    Applies minimum token penalty by setting the logits of the stop tokens
    to -inf.
    """
    min_tokens_logits_to_penalize: List[Tuple[int, int]] = []
    for index, min_token in enumerate(min_tokens):
        if (len(output_token_ids[index]) < min_token):
            for stop_token_id in stop_token_ids[index]:
                min_tokens_logits_to_penalize.append((index, stop_token_id))
    if min_tokens_logits_to_penalize:
        logits[tuple(zip(*min_tokens_logits_to_penalize))] = -float("inf")


def _apply_penalties(logits: torch.Tensor, prompt_token_ids: torch.Tensor,
                     presence_penalties: torch.Tensor,
                     frequency_penalties: torch.Tensor,
                     repetition_penalties: torch.Tensor,
                     output_token_ids: List[List[int]]):
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size,
                                          logits.device)
    return apply_penalties(logits, prompt_token_ids, output_tokens_t,
                           presence_penalties, frequency_penalties,
                           repetition_penalties)


def _convert_to_tensors(output_token_ids: List[List[int]], vocab_size: int,
                        device: torch.device) -> torch.Tensor:
    """
    Convert the different list data structures to tensors.
    """
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    )
    return output_tokens_tensor.to(device, non_blocking=True)
