# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@dataclass
class SamplingMetadata:
    temperature: torch.Tensor

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None
    min_p: torch.Tensor | None

    repetition_penalty: torch.Tensor
    frequency_penalty: torch.Tensor
    presence_penalty: torch.Tensor

    seeds: torch.Tensor
    pos: torch.Tensor

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    # For penalties
    idx_mapping: torch.Tensor
    prompt_bin_mask: torch.Tensor
    output_bin_counts: torch.Tensor

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        device: torch.device,
    ) -> "SamplingMetadata":
        assert num_reqs > 0
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
        seeds = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        pos = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        max_num_logprobs = 20

        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
        # NOTE(woosuk): These are placeholder tensors to avoid None checks in the
        # penalties kernel. We use 2 instead of 1 as vocab_size to avoid Triton
        # specialization and re-compilation at runtime.
        prompt_bin_mask = torch.zeros(num_reqs, 2, dtype=torch.int32, device=device)
        output_bin_counts = torch.zeros(num_reqs, 2, dtype=torch.int32, device=device)

        return cls(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seeds=seeds,
            pos=pos,
            max_num_logprobs=max_num_logprobs,
            idx_mapping=idx_mapping,
            prompt_bin_mask=prompt_bin_mask,
            output_bin_counts=output_bin_counts,
        )


# NOTE(woosuk): Re-compilation can happen at runtime since top_p and top_k can be None.
@triton.jit
def _expand_sampling_metadata_kernel(
    temp_ptr,
    expanded_temp_ptr,
    top_p_ptr,
    expanded_top_p_ptr,
    top_k_ptr,
    expanded_top_k_ptr,
    min_p_ptr,
    expanded_min_p_ptr,
    rep_penalty_ptr,
    expanded_rep_penalty_ptr,
    freq_penalty_ptr,
    expanded_freq_penalty_ptr,
    pres_penalty_ptr,
    expanded_pres_penalty_ptr,
    seeds_ptr,
    expanded_seeds_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens

    temp = tl.load(temp_ptr + req_idx)
    tl.store(expanded_temp_ptr + start_idx + block, temp, mask=mask)

    if top_p_ptr is not None:
        top_p = tl.load(top_p_ptr + req_idx)
        tl.store(expanded_top_p_ptr + start_idx + block, top_p, mask=mask)

    if top_k_ptr is not None:
        top_k = tl.load(top_k_ptr + req_idx)
        tl.store(expanded_top_k_ptr + start_idx + block, top_k, mask=mask)

    if min_p_ptr is not None:
        min_p = tl.load(min_p_ptr + req_idx)
        tl.store(expanded_min_p_ptr + start_idx + block, min_p, mask=mask)

    rep_penalty = tl.load(rep_penalty_ptr + req_idx)
    tl.store(expanded_rep_penalty_ptr + start_idx + block, rep_penalty, mask=mask)

    freq_penalty = tl.load(freq_penalty_ptr + req_idx)
    tl.store(expanded_freq_penalty_ptr + start_idx + block, freq_penalty, mask=mask)

    pres_penalty = tl.load(pres_penalty_ptr + req_idx)
    tl.store(expanded_pres_penalty_ptr + start_idx + block, pres_penalty, mask=mask)

    seed = tl.load(seeds_ptr + req_idx)
    tl.store(expanded_seeds_ptr + start_idx + block, seed, mask=mask)


def expand_sampling_metadata(
    sampling_metadata: SamplingMetadata,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> SamplingMetadata:
    total_num_logits = sampling_metadata.pos.shape[0]
    create_empty = lambda x: x.new_empty(total_num_logits) if x is not None else None
    expanded_temp = create_empty(sampling_metadata.temperature)
    expanded_top_p = create_empty(sampling_metadata.top_p)
    expanded_top_k = create_empty(sampling_metadata.top_k)
    expanded_min_p = create_empty(sampling_metadata.min_p)
    expanded_repetition_penalty = create_empty(sampling_metadata.repetition_penalty)
    expanded_frequency_penalty = create_empty(sampling_metadata.frequency_penalty)
    expanded_presence_penalty = create_empty(sampling_metadata.presence_penalty)
    expanded_seeds = create_empty(sampling_metadata.seeds)

    num_reqs = cu_num_logits.shape[0] - 1
    _expand_sampling_metadata_kernel[(num_reqs,)](
        sampling_metadata.temperature,
        expanded_temp,
        sampling_metadata.top_p,
        expanded_top_p,
        sampling_metadata.top_k,
        expanded_top_k,
        sampling_metadata.min_p,
        expanded_min_p,
        sampling_metadata.repetition_penalty,
        expanded_repetition_penalty,
        sampling_metadata.frequency_penalty,
        expanded_frequency_penalty,
        sampling_metadata.presence_penalty,
        expanded_presence_penalty,
        sampling_metadata.seeds,
        expanded_seeds,
        cu_num_logits,
        BLOCK_SIZE=triton.next_power_of_2(max_expand_len),
    )
    return SamplingMetadata(
        temperature=expanded_temp,
        top_p=expanded_top_p,
        top_k=expanded_top_k,
        min_p=expanded_min_p,
        seeds=expanded_seeds,
        repetition_penalty=expanded_repetition_penalty,
        frequency_penalty=expanded_frequency_penalty,
        presence_penalty=expanded_presence_penalty,
        pos=sampling_metadata.pos,
        max_num_logprobs=sampling_metadata.max_num_logprobs,
        # TODO(woosuk): Support penalties with spec decoding.
        idx_mapping=sampling_metadata.idx_mapping,
        prompt_bin_mask=sampling_metadata.prompt_bin_mask,
        output_bin_counts=sampling_metadata.output_bin_counts,
    )
