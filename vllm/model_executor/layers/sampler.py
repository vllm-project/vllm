"""A layer that samples the next tokens from the model's outputs."""
from dataclasses import dataclass
from math import inf
from typing import Dict, List, Optional, Tuple, Union

import msgspec
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.sampler_output import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        logits = self.apply_penalties(logits, sampling_metadata)

        probs = self.get_probs(logits)
        sampled = self.sample(probs, sampling_metadata)

        if sampling_metadata.max_num_logprobs > 0:
            logprobs = self.get_logprobs(logits)
            topk_logprobs, topk_indices = torch.topk(
                logprobs, sampling_metadata.max_num_logprobs, dim=-1)
        else:
            topk_logprobs = None
            topk_indices = None

        sampler_output = SamplerOutput(
            sampled_token_ids=sampled,
            logprob_token_ids=topk_indices,
            logprobs=topk_logprobs,
            prompt_logprob_token_ids=None,
            prompt_logprobs=None,
            model_forward_time=0.0,
            model_execute_time=0.0,
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

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return logits

    def get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1, dtype=torch.float32)

    def get_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(logits, dim=-1, dtype=torch.float32)

    def greedy_sample(self, probs: torch.Tensor) -> torch.Tensor:
        return probs.argmax(dim=1).view(-1)

    def random_sample(
        self,
        probs: torch.Tensor,
        generators: Optional[List[torch.Generator]],
        no_generator: bool,
    ) -> torch.Tensor:
        q = torch.empty_like(probs)
        if no_generator:
            q.exponential_()
        else:
            assert generators is not None and len(generators) == probs.shape[0]
            # TODO(woosuk): Optimize this.
            for i, generator in enumerate(generators):
                q[i].exponential_(generator=generator)
        return probs.div_(q).argmax(dim=1).view(-1)

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
            return self.random_sample(probs, sampling_metadata.generators,
                                      sampling_metadata.no_generator)

        greedy_sampled = self.greedy_sample(probs)
        random_sampled = self.random_sample(probs,
                                            sampling_metadata.generators,
                                            sampling_metadata.no_generator)
        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
        )
        return sampled
