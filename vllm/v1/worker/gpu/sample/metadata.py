# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch


@dataclass
class SamplingMetadata:
    idx_mapping: torch.Tensor

    temperature: torch.Tensor

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None
    min_p: torch.Tensor | None

    # For penalties
    repetition_penalty: torch.Tensor
    frequency_penalty: torch.Tensor
    presence_penalty: torch.Tensor
    prompt_bin_mask: torch.Tensor
    output_bin_counts: torch.Tensor

    seeds: torch.Tensor
    pos: torch.Tensor

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        device: torch.device,
    ) -> "SamplingMetadata":
        assert num_reqs > 0
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)

        temperature = torch.zeros(num_reqs, dtype=torch.float32, device=device)
        temperature[0] = 0.5
        # TODO(woosuk): Use top-p and top-k for dummy sampler.
        # Currently, they are disabled because of memory usage.
        # top_p = torch.full((num_reqs,), 0.95, dtype=torch.float32, device=device)
        # top_k = torch.full((num_reqs,), 20, dtype=torch.int32, device=device)
        top_p = None
        top_k = None
        min_p = torch.zeros(num_reqs, dtype=torch.float32, device=device)
        # NOTE(woosuk): We must set penalties to their default values to make sure
        # the penalties kernel does not touch the placeholder bin_counts tensors.
        repetition_penalty = torch.ones(num_reqs, dtype=torch.float32, device=device)
        frequency_penalty = torch.zeros(num_reqs, dtype=torch.float32, device=device)
        presence_penalty = torch.zeros(num_reqs, dtype=torch.float32, device=device)

        # NOTE(woosuk): These are placeholder tensors to avoid None checks in the
        # penalties kernel. We use 2 instead of 1 as vocab_size to avoid Triton
        # specialization and re-compilation at runtime.
        prompt_bin_mask = torch.zeros(num_reqs, 2, dtype=torch.int32, device=device)
        output_bin_counts = torch.zeros(num_reqs, 2, dtype=torch.int32, device=device)

        seeds = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        pos = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        max_num_logprobs = 20

        return cls(
            idx_mapping=idx_mapping,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            prompt_bin_mask=prompt_bin_mask,
            output_bin_counts=output_bin_counts,
            seeds=seeds,
            pos=pos,
            max_num_logprobs=max_num_logprobs,
        )
