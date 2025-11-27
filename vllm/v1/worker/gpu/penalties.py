# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


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


def apply_penalties(
    logits: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    idx_mapping: torch.Tensor,
    prompt_bin_counts: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _penalties_kernel[(num_reqs, num_blocks)](
        logits,
        logits.stride(0),
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        idx_mapping,
        prompt_bin_counts,
        prompt_bin_counts.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
