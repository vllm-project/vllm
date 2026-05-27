# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MTP-input RMSNorm: enorm (with mask-zero at position 0) + hnorm.

Replaces the eager sequence at the top of the MTP draft forward:
    inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    inputs_embeds = self.enorm(inputs_embeds)
    previous_hidden_states = previous_hidden_states.view(-1, hc_mult, H)
    previous_hidden_states = self.hnorm(previous_hidden_states)

which lowers to ~6 small kernels (CompareEq, where, Fill, enorm rms_norm,
hnorm rms_norm, plus aten elementwise helpers) on the breakable-cudagraph
path. Math is preserved: positions==0 → masked row → zero RMS output
regardless of weight.

A single grid (T, hc_mult+1) drives both norms: task 0 is enorm on
inputs_embeds[token, :], task k+1 is hnorm on previous_hidden_states[token, k, :].
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rmsnorm_row(
    x,
    w_ptr,
    out_row_ptr,
    block,
    mask,
    eps,
    HIDDEN: tl.constexpr,
):
    x = x.to(tl.float32)
    variance = tl.sum(x * x, axis=0) / HIDDEN
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(w_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(out_row_ptr + block, y.to(out_row_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_mtp_input_rmsnorm_kernel(
    inputs_embeds_ptr,
    positions_ptr,
    prev_hidden_ptr,
    enorm_weight_ptr,
    hnorm_weight_ptr,
    enorm_out_ptr,
    hnorm_out_ptr,
    eps,
    HIDDEN: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # int64 token index so per-token offsets don't overflow int32 at
    # large num_tokens (matches the convention in fused_q_kv_rmsnorm).
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < HIDDEN

    if pid_task == 0:
        # enorm path: load inputs_embeds[token, :] then zero-mask at pos==0.
        # Math is preserved: pos==0 → x=0 → variance=0 → RMSNorm output is 0
        # regardless of weight, matching torch.where(pos==0, 0, x) + RMSNorm.
        pos = tl.load(positions_ptr + token_idx)
        keep = pos != 0
        x = tl.load(
            inputs_embeds_ptr + token_idx * HIDDEN + block, mask=mask, other=0.0
        )
        x = tl.where(keep, x, 0.0)
        _rmsnorm_row(
            x,
            enorm_weight_ptr,
            enorm_out_ptr + token_idx * HIDDEN,
            block,
            mask,
            eps,
            HIDDEN,
        )
    else:
        # hnorm path: load prev_hidden[token, slot, :].
        slot = pid_task - 1
        row_offset = (token_idx * HC_MULT + slot) * HIDDEN
        x = tl.load(prev_hidden_ptr + row_offset + block, mask=mask, other=0.0)
        _rmsnorm_row(
            x,
            hnorm_weight_ptr,
            hnorm_out_ptr + row_offset,
            block,
            mask,
            eps,
            HIDDEN,
        )


def fused_mtp_input_rmsnorm(
    inputs_embeds: torch.Tensor,
    positions: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
    hc_mult: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (enorm_out, hnorm_out).

    enorm_out has the same shape as inputs_embeds (2D, [T, H]).
    hnorm_out has the same shape as previous_hidden_states (3D, [T, hc_mult, H]).
    previous_hidden_states must already be reshaped to 3D.
    """
    assert inputs_embeds.ndim == 2
    assert previous_hidden_states.ndim == 3
    assert previous_hidden_states.shape[1] == hc_mult
    assert inputs_embeds.shape[0] == previous_hidden_states.shape[0], (
        "token dim mismatch"
    )
    assert (
        inputs_embeds.shape[1]
        == previous_hidden_states.shape[2]
        == enorm_weight.shape[0]
        == hnorm_weight.shape[0]
    )
    assert inputs_embeds.is_contiguous() and previous_hidden_states.is_contiguous()
    assert enorm_weight.is_contiguous() and hnorm_weight.is_contiguous()

    num_tokens, hidden = inputs_embeds.shape
    enorm_out = torch.empty_like(inputs_embeds)
    hnorm_out = torch.empty_like(previous_hidden_states)
    if num_tokens == 0:
        return enorm_out, hnorm_out

    block_size = triton.next_power_of_2(hidden)
    _fused_mtp_input_rmsnorm_kernel[(num_tokens, hc_mult + 1)](
        inputs_embeds,
        positions,
        previous_hidden_states,
        enorm_weight,
        hnorm_weight,
        enorm_out,
        hnorm_out,
        eps,
        HIDDEN=hidden,
        HC_MULT=hc_mult,
        BLOCK_SIZE=block_size,
    )
    return enorm_out, hnorm_out
