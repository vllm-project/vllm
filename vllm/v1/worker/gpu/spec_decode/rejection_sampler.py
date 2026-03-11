# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS


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
    sampled = target_sampled.new_empty(num_reqs, num_speculative_steps + 1)
    num_sampled = target_sampled.new_empty(num_reqs, dtype=torch.int32)
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
def _gather_draft_logits_and_target_argmax_kernel(
    local_target_argmax_ptr,
    local_target_argmax_stride,
    local_target_max_ptr,
    local_target_max_stride,
    # [num_logits, V]
    out_draft_logits_ptr,
    out_draft_logits_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
):
    logit_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    if temp == 0.0:
        # Greedy sampling. Get the target logits argmax.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(
            local_target_argmax_ptr
            + logit_idx * local_target_argmax_stride
            + block_idx,
            token_id,
        )
        tl.store(
            local_target_max_ptr + logit_idx * local_target_max_stride + block_idx,
            value,
        )
    elif draft_step_idx < num_speculative_steps:
        draft_logits = tl.load(
            draft_logits_ptr
            + req_state_idx * draft_logits_stride_0
            + draft_step_idx * draft_logits_stride_1
            + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        tl.store(
            out_draft_logits_ptr + logit_idx * out_draft_logits_stride + block_offsets,
            draft_logits,
            mask=mask,
        )


@triton.jit
def _probabilistic_rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_reqs]
    rejected_pos_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits, V]
    target_probs_ptr,
    target_probs_stride,
    # [num_logits, V]
    draft_probs_ptr,
    draft_probs_stride,
    # [num_logits, num_blocks]
    local_target_argmax_ptr,
    local_target_argmax_stride,
    # [num_logits, num_blocks]
    local_target_max_ptr,
    local_target_max_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    pos_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seeds_ptr,
    NUM_BLOCKS: tl.constexpr,
    PADDED_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_tokens = tl.load(cu_num_logits_ptr + req_idx + 1) - start_idx
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    seed = tl.load(seeds_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    rejected_step = 0
    accepted = True
    for i in range(num_tokens - 1):
        if accepted:
            logit_idx = start_idx + i
            draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1)
            if temp == 0.0:
                # Greedy sampling. Only accept the sampled draft token if
                # it exactly matches the target argmax.
                block_offsets = tl.arange(0, PADDED_NUM_BLOCKS)
                block_mask = block_offsets < NUM_BLOCKS
                local_max = tl.load(
                    local_target_max_ptr
                    + logit_idx * local_target_max_stride
                    + block_offsets,
                    mask=block_mask,
                    other=float("-inf"),
                )
                max_block = tl.argmax(local_max, axis=0)
                target_argmax = tl.load(
                    local_target_argmax_ptr
                    + logit_idx * local_target_argmax_stride
                    + max_block
                )
                accepted &= target_argmax == draft_sampled
            else:
                target_prob = tl.load(
                    target_probs_ptr + logit_idx * target_probs_stride + draft_sampled
                )
                draft_prob = tl.load(
                    draft_probs_ptr + logit_idx * draft_probs_stride + draft_sampled
                )
                pos = tl.load(pos_ptr + logit_idx)
                u = tl.sum(tl.rand(seed, pos + tl.arange(0, 1)))
                accepted &= target_prob > u * draft_prob
            tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            rejected_step += accepted
    tl.store(rejected_steps_ptr + req_idx, rejected_step)
    pos_val = tl.load(pos_ptr + start_idx + rejected_step)
    tl.store(rejected_pos_ptr + req_idx, pos_val)


@triton.jit
def _compute_residual_logits_kernel(
    # [num_reqs, V]
    residual_logits_ptr,
    residual_logits_stride,
    # [num_logits, V]
    target_probs_ptr,
    target_probs_stride,
    # [num_logits, V]
    draft_probs_ptr,
    draft_probs_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_reqs]
    rejected_step_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    rejected_logit_idx = start_idx + tl.load(rejected_step_ptr + req_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if temp == 0.0 or (rejected_logit_idx == end_idx - 1):
        # Greedy sampling / bonus token. In either case, use the
        # target logits directly to reduce numerical error.
        residual_logits = tl.load(
            target_logits_ptr
            + rejected_logit_idx * target_logits_stride
            + block_offsets,
            mask=mask,
            other=float("-inf"),
        )
    else:
        target_probs = tl.load(
            target_probs_ptr + rejected_logit_idx * target_probs_stride + block_offsets,
            mask=mask,
            other=0.0,
        )
        draft_probs = tl.load(
            draft_probs_ptr + rejected_logit_idx * draft_probs_stride + block_offsets,
            mask=mask,
            other=0.0,
        )
        residual_probs = tl.maximum(target_probs - draft_probs, 0.0)
        residual_logits = tl.log(residual_probs)

    tl.store(
        residual_logits_ptr + req_idx * residual_logits_stride + block_offsets,
        residual_logits,
        mask=mask,
    )


def probabilistic_rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)

    # Gather draft logits and target argmax for greedy sampling.
    gathered_draft_logits = target_logits.new_empty(target_logits.shape)
    local_target_argmax = target_logits.new_empty(
        num_logits, num_blocks, dtype=torch.int64
    )
    local_target_max = target_logits.new_empty(
        num_logits, num_blocks, dtype=torch.float32
    )
    _gather_draft_logits_and_target_argmax_kernel[(num_logits, num_blocks)](
        local_target_argmax,
        local_target_argmax.stride(0),
        local_target_max,
        local_target_max.stride(0),
        gathered_draft_logits,
        gathered_draft_logits.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Compute target and draft probs.
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_probs = torch.softmax(gathered_draft_logits, dim=-1)

    # Rejection sample.
    # [num_reqs, num_speculative_steps + 1]
    sampled = draft_sampled.new_zeros(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    # [num_reqs]
    rejected_steps = sampled.new_empty(num_reqs)
    # [num_reqs]
    rejected_pos = pos.new_empty(num_reqs)
    _probabilistic_rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        rejected_steps,
        rejected_pos,
        draft_sampled,
        target_probs,
        target_probs.stride(0),
        draft_probs,
        draft_probs.stride(0),
        local_target_argmax,
        local_target_argmax.stride(0),
        local_target_max,
        local_target_max.stride(0),
        cu_num_logits,
        pos,
        idx_mapping,
        temperature,
        seed,
        num_warps=1,
        NUM_BLOCKS=num_blocks,
        PADDED_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
    )

    # Compute the logits and positions to resample the rejected/bonus
    # tokens from.
    # [num_reqs, vocab_size]
    residual_logits = target_logits.new_empty(num_reqs, vocab_size)
    _compute_residual_logits_kernel[(num_reqs, num_blocks)](
        residual_logits,
        residual_logits.stride(0),
        target_probs,
        target_probs.stride(0),
        draft_probs,
        draft_probs.stride(0),
        target_logits,
        target_logits.stride(0),
        rejected_steps,
        cu_num_logits,
        idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Gumbel sample tokens from the residual distribution.
    resampled = gumbel_sample(
        residual_logits,
        idx_mapping,
        temperature,
        seed,
        rejected_pos,
        apply_temperature=False,
    )
    sampled.scatter_(1, rejected_steps.unsqueeze(1), resampled.unsqueeze(1))

    return sampled, rejected_steps + 1


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    for i in range(num_tokens):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


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

    def _get_logprobs_tensors(
        self,
        input_batch: InputBatch,
        sampled: torch.Tensor,
        logits: torch.Tensor,
    ) -> LogprobsTensors | None:
        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = input_batch.cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.empty(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            input_batch.cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != input_batch.idx_mapping.shape[0]
        return compute_topk_logprobs(
            logits,
            max_num_logprobs,
            flat_sampled,
            input_batch.cu_num_logits_np.tolist() if expanded_logits else None,
        )

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
            sampler_output = self.sampler(logits, input_batch)
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
            sampled, num_sampled = probabilistic_rejection_sample(
                processed_logits,
                draft_logits,
                draft_sampled,
                input_batch.cu_num_logits,
                pos,
                input_batch.idx_mapping,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                self.num_speculative_steps,
            )
            logprobs_tensors = self._get_logprobs_tensors(
                input_batch,
                sampled,
                processed_logits
                if self.sampler.logprobs_mode == "processed_logprobs"
                else logits,
            )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
        )
