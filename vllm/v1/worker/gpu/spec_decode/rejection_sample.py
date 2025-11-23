# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.outputs import SamplerOutput
from vllm.v1.worker.gpu.sampler import Sampler
from vllm.v1.worker.gpu.states import SamplingMetadata

PLACEHOLDER_TOKEN_ID = -1


class RejectionSampler:
    def __init__(self, sampler: Sampler, num_speculative_steps: int):
        self.sampler = sampler
        self.num_speculative_steps = num_speculative_steps

    def __call__(
        self,
        # [num_draft_tokens + num_reqs, vocab_size]
        logits: torch.Tensor,
        # [num_reqs]
        sampling_metadata: SamplingMetadata,
        # [num_draft_tokens + num_reqs]
        input_ids: torch.Tensor,
        # [num_reqs + 1]
        cu_num_logits: torch.Tensor,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        num_draft_tokens = num_logits - num_reqs
        assert num_draft_tokens > 0, "No draft tokens"

        sampling_metadata = expand_sampling_metadata(
            sampling_metadata, cu_num_logits, num_logits, self.num_speculative_steps
        )
        target_sampled, _ = self.sampler.sample(
            logits, sampling_metadata, return_logits=False
        )

        sampled = torch.empty(
            num_reqs,
            self.num_speculative_steps + 1,
            dtype=torch.int64,
            device=logits.device,
        )
        num_sampled = torch.empty(
            num_reqs,
            dtype=torch.int32,
            device=logits.device,
        )
        last_sampled = torch.empty(
            num_reqs,
            1,
            dtype=torch.int64,
            device=logits.device,
        )
        _rejection_sample_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            last_sampled,
            target_sampled,
            input_ids,
            cu_num_logits,
            num_warps=1,
        )

        # For the compatibility with the SamplerOutput.
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
        )
        return sampler_output, num_sampled, last_sampled


@triton.jit
def _rejection_sample_kernel(
    sampled_ptr,  # [num_reqs, num_speculative_steps + 1]
    sampled_stride,
    num_sampled_ptr,  # [num_reqs]
    last_sampled_ptr,  # [num_reqs]
    target_sampled_ptr,  # [num_draft_tokens + num_reqs]
    input_ids_ptr,  # [num_draft_tokens + num_reqs]
    cu_num_logits_ptr,  # [num_reqs + 1]
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    num_sampled = 0
    last_sampled = 0
    rejected = False
    for i in range(num_tokens):
        if not rejected:
            target_sampled = tl.load(target_sampled_ptr + start_idx + i)
            draft_sampled = tl.load(
                input_ids_ptr + start_idx + i + 1, mask=i < num_tokens - 1, other=0
            )
            tl.store(sampled_ptr + req_idx * sampled_stride + i, target_sampled)
            num_sampled += 1
            last_sampled = target_sampled
            if target_sampled != draft_sampled:
                rejected = True
    tl.store(num_sampled_ptr + req_idx, num_sampled)
    tl.store(last_sampled_ptr + req_idx, last_sampled)


# NOTE(woosuk): Re-compilation can happen due to top_p and top_k.
@triton.jit
def _expand_kernel(
    temp_ptr,
    expanded_temp_ptr,
    top_p_ptr,
    expanded_top_p_ptr,
    top_k_ptr,
    expanded_top_k_ptr,
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

    seed = tl.load(seeds_ptr + req_idx)
    tl.store(expanded_seeds_ptr + start_idx + block, seed, mask=mask)


def expand_sampling_metadata(
    sampling_metadata: SamplingMetadata,
    cu_num_logits: torch.Tensor,
    num_logits: int,
    num_speculative_steps: int,
) -> SamplingMetadata:
    create_empty = lambda x: x.new_empty(num_logits) if x is not None else None
    expanded_temp = create_empty(sampling_metadata.temperature)
    expanded_top_p = create_empty(sampling_metadata.top_p)
    expanded_top_k = create_empty(sampling_metadata.top_k)
    expanded_seeds = create_empty(sampling_metadata.seeds)

    num_reqs = cu_num_logits.shape[0] - 1
    _expand_kernel[(num_reqs,)](
        sampling_metadata.temperature,
        expanded_temp,
        sampling_metadata.top_p,
        expanded_top_p,
        sampling_metadata.top_k,
        expanded_top_k,
        sampling_metadata.seeds,
        expanded_seeds,
        cu_num_logits,
        BLOCK_SIZE=triton.next_power_of_two(num_speculative_steps + 1),
    )
    return SamplingMetadata(
        temperature=expanded_temp,
        top_p=expanded_top_p,
        top_k=expanded_top_k,
        seeds=expanded_seeds,
        pos=sampling_metadata.pos,
        max_num_logprobs=-1,
    )
