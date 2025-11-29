# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.metadata import SamplingMetadata


@triton.jit
def _penalties_kernel(
    logits_ptr,
    logits_stride,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    idx_mapping_ptr,
    prompt_bin_counts_ptr,
    prompt_bin_counts_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    rep_penalty = tl.load(repetition_penalty_ptr + batch_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + batch_idx)
    pres_penalty = tl.load(presence_penalty_ptr + batch_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    if not (use_rep_penalty or use_freq_penalty or use_pres_penalty):
        # No penalties to apply. Early return.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + batch_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    output_bin_counts = tl.load(
        output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
        mask=mask,
    )

    # Apply repetition penalties.
    if use_rep_penalty:
        prompt_bin_counts = tl.load(
            prompt_bin_counts_ptr + req_state_idx * prompt_bin_counts_stride + block,
            mask=mask,
        )
        # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
        scale = tl.where((prompt_bin_counts + output_bin_counts) > 0, rep_penalty, 1.0)
        # If logits are positive, divide by penalty, otherwise multiply by penalty.
        scale = tl.where(logits > 0, 1.0 / scale, scale)
        logits *= scale

    # Apply frequency penalties.
    logits -= freq_penalty * output_bin_counts
    # Apply presence penalties.
    logits -= pres_penalty * (output_bin_counts > 0)
    # Store back to logits.
    tl.store(logits_ptr + batch_idx * logits_stride + block, logits, mask=mask)


def apply_penalties(logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> None:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _penalties_kernel[(num_reqs, num_blocks)](
        logits,
        logits.stride(0),
        sampling_metadata.repetition_penalty,
        sampling_metadata.frequency_penalty,
        sampling_metadata.presence_penalty,
        sampling_metadata.idx_mapping,
        sampling_metadata.prompt_bin_counts,
        sampling_metadata.prompt_bin_counts.stride(0),
        sampling_metadata.output_bin_counts,
        sampling_metadata.output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit(do_not_specialize=["prefill_len", "prompt_len"])
def _bincount_kernel(
    prefill_token_ids_ptr,
    prefill_len,
    prompt_len,
    prompt_bin_counts_ptr,
    output_bin_counts_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if block_idx * BLOCK_SIZE >= prefill_len:
        return

    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if block_idx * BLOCK_SIZE < prompt_len:
        mask = block < prompt_len
        prefill_tokens = tl.load(prefill_token_ids_ptr + block, mask=mask)
        tl.atomic_add(prompt_bin_counts_ptr + prefill_tokens, 1, mask=mask)
    if (block_idx + 1) * BLOCK_SIZE >= prompt_len:
        mask = block < prefill_len
        mask &= block >= prompt_len
        prefill_tokens = tl.load(prefill_token_ids_ptr + block, mask=mask)
        tl.atomic_add(output_bin_counts_ptr + prefill_tokens, 1, mask=mask)


def bincount(
    prefill_token_ids: torch.Tensor,
    prefill_len: int,
    prompt_len: int,
    prompt_bin_counts: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    prompt_bin_counts.zero_()
    output_bin_counts.zero_()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(prefill_len, BLOCK_SIZE)
    _bincount_kernel[(num_blocks,)](
        prefill_token_ids,
        prefill_len,
        prompt_len,
        prompt_bin_counts,
        output_bin_counts,
        BLOCK_SIZE=BLOCK_SIZE,
    )
