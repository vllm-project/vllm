# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler


@triton.jit
def _strict_rejection_sample_kernel(
    sampled_ptr,  # [num_reqs, num_speculative_steps + 1]
    sampled_stride,
    num_sampled_ptr,  # [num_reqs]
    target_sampled_ptr,  # [num_draft_tokens + num_reqs]
    input_ids_ptr,  # [num_draft_tokens + num_reqs]
    cu_num_logits_ptr,  # [num_reqs + 1]
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    num_sampled = 0
    rejected = False
    for i in range(num_tokens - 1):
        if not rejected:
            target_sampled = tl.load(target_sampled_ptr + start_idx + i)
            draft_sampled = tl.load(input_ids_ptr + start_idx + i + 1)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, target_sampled)
            num_sampled += 1
            if target_sampled != draft_sampled:
                rejected = True
    if not rejected:
        target_sampled = tl.load(target_sampled_ptr + start_idx + num_tokens - 1)
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_tokens - 1, target_sampled
        )
        num_sampled += 1
    tl.store(num_sampled_ptr + req_idx, num_sampled)


def strict_rejection_sample(
    # [num_draft_tokens + num_reqs]
    target_sampled: torch.Tensor,
    # [num_draft_tokens + num_reqs]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    num_speculative_steps,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = torch.empty(
        num_reqs,
        num_speculative_steps + 1,
        dtype=target_sampled.dtype,
        device=target_sampled.device,
    )
    num_sampled = torch.empty(
        num_reqs,
        dtype=torch.int32,
        device=target_sampled.device,
    )
    _strict_rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_sampled,
        draft_sampled,
        cu_num_logits,
        num_warps=1,
    )
    return sampled, num_sampled


@triton.jit
def _probabilistic_rejection_sample_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits, V]
    target_probs_ptr,
    target_probs_stride,
    # [num_reqs, num_speculative_steps, V]
    draft_probs_ptr,
    draft_probs_stride_0,
    draft_probs_stride_1,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    pos_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [num_reqs]
    seeds_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_tokens = tl.load(cu_num_logits_ptr + req_idx + 1) - start_idx
    seed = tl.load(seeds_ptr + tl.load(idx_mapping_ptr + req_idx))

    rejected_step = 0
    accepted = True
    for i in range(num_tokens - 1):
        if accepted:
            draft_sampled = tl.load(draft_sampled_ptr + start_idx + i + 1)
            target_prob = tl.load(
                target_probs_ptr + (start_idx + i) * target_probs_stride + draft_sampled
            )
            draft_prob = tl.load(
                draft_probs_ptr
                + req_idx * draft_probs_stride_0
                + i * draft_probs_stride_1
                + draft_sampled
            )
            pos = tl.load(pos_ptr + start_idx + i)
            u = tl.sum(tl.rand(seed, pos + tl.arange(0, 1)))
            accepted &= target_prob > u * draft_prob
            tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            rejected_step += accepted
    tl.store(rejected_steps_ptr + req_idx, rejected_step)


@triton.jit
def _compute_residual_logits_kernel(
    # [num_reqs, V]
    residual_logits_ptr,
    residual_logits_stride,
    # [num_reqs]
    residual_pos_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, V]
    target_probs_ptr,
    target_probs_stride,
    # [num_reqs, num_speculative_steps, V]
    draft_probs_ptr,
    draft_probs_stride_0,
    draft_probs_stride_1,
    # [num_reqs]
    rejected_step_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    rejected_draft_step = tl.load(rejected_step_ptr + req_idx)
    rejected_logit_idx = start_idx + rejected_draft_step

    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if rejected_logit_idx < end_idx - 1:
        target_probs = tl.load(
            target_probs_ptr + rejected_logit_idx * target_probs_stride + block_offsets,
            mask=mask,
            other=0.0,
        )
        draft_probs = tl.load(
            draft_probs_ptr
            + req_idx * draft_probs_stride_0
            + rejected_draft_step * draft_probs_stride_1
            + block_offsets,
            mask=mask,
            other=0.0,
        )
        residual_probs = tl.maximum(target_probs - draft_probs, 0.0)
        residual_logits = tl.log(residual_probs)
    else:
        # This is a bonus token. Directly return the target logits.
        residual_logits = tl.load(
            target_logits_ptr
            + rejected_logit_idx * target_logits_stride
            + block_offsets,
            mask=mask,
            other=0.0,
        )

    tl.store(
        residual_logits_ptr + req_idx * residual_logits_stride + block_offsets,
        residual_logits,
        mask=mask,
    )

    # First block computes the residual logit positions.
    if block_idx == 0:
        pos_val = tl.load(pos_ptr + rejected_logit_idx)
        tl.store(residual_pos_ptr + req_idx, pos_val)


def probabilistic_rejection_sample(
    # [num_draft_tokens + num_reqs, V]
    target_logits: torch.Tensor,
    # [num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor,
    # [num_draft_tokens + num_reqs]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    temperature,
    seeds,
    num_speculative_steps,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    device = target_logits.device
    vocab_size = target_logits.shape[-1]

    # Compute target and draft probs.
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_probs = torch.softmax(draft_logits, dim=-1)

    # Rejection sample.
    # [num_reqs, num_speculative_steps + 1]
    sampled = torch.empty(
        num_reqs,
        num_speculative_steps + 1,
        dtype=torch.int64,
        device=device,
    )
    # [num_reqs]
    rejected_steps = torch.empty(
        num_reqs,
        dtype=torch.int64,
        device=device,
    )
    _probabilistic_rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        rejected_steps,
        draft_sampled,
        target_probs,
        target_probs.stride(0),
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        cu_num_logits,
        pos,
        idx_mapping,
        seeds,
        num_warps=1,
    )

    # Compute the logits and positions to resample the rejected/bonus
    # tokens from.
    # [num_reqs, vocab_size]
    residual_logits = torch.empty(
        num_reqs,
        vocab_size,
        dtype=target_logits.dtype,
        device=device,
    )
    # [num_reqs]
    residual_pos = torch.empty(
        num_reqs,
        dtype=pos.dtype,
        device=device,
    )
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _compute_residual_logits_kernel[(num_reqs, num_blocks)](
        residual_logits,
        residual_logits.stride(0),
        residual_pos,
        target_logits,
        target_logits.stride(0),
        target_probs,
        target_probs.stride(0),
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        rejected_steps,
        cu_num_logits,
        pos,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Gumbel sample tokens from the residual distribution.
    resampled = gumbel_sample(
        residual_logits,
        idx_mapping,
        temperature,
        seeds,
        residual_pos,
        apply_temperature=False,
    )
    sampled.scatter_(1, rejected_steps.unsqueeze(1), resampled.unsqueeze(1))

    return sampled, rejected_steps + 1


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        num_speculative_steps,
        use_strict_rejection_sampling: bool = True,
    ):
        self.sampler = sampler
        self.num_speculative_steps = num_speculative_steps
        self.use_strict_rejection_sampling = use_strict_rejection_sampling

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        if self.use_strict_rejection_sampling:
            sampler_output = self.sampler(
                logits,
                input_batch,
            )
            logprobs_tensors = sampler_output.logprobs_tensors
            sampled, num_sampled = strict_rejection_sample(
                sampler_output.sampled_token_ids.view(-1),
                draft_sampled,
                input_batch.cu_num_logits,
                self.num_speculative_steps,
            )
        else:
            assert draft_logits is not None
            pos = input_batch.positions[input_batch.logits_indices]
            processed_logits = self.sampler.apply_sampling_params(
                logits,
                input_batch.expanded_idx_mapping,
                input_batch.idx_mapping_np,
                pos,
                draft_sampled,
                input_batch.expanded_local_pos,
            )
            # TODO (TheEpicDolphin): Return logprobs for sampled token ids.
            logprobs_tensors = None
            sampled, num_sampled = probabilistic_rejection_sample(
                processed_logits,
                draft_logits,
                draft_sampled,
                input_batch.cu_num_logits,
                pos,
                input_batch.idx_mapping,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                self.num_speculative_steps,
            )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
        )
