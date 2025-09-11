# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.triton_utils import tl, triton


@triton.jit
def _correct_attn_cp_out_kernel(outputs_ptr, new_output_ptr, lses_ptr,
                                vlse_ptr, outputs_stride_B, outputs_stride_H,
                                outputs_stride_D, lses_stride_N, lses_stride_B,
                                lses_stride_H, lse_idx, HEAD_DIM: tl.constexpr,
                                N_ROUNDED: tl.constexpr):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        output: [ B, H, D ]
        lses   : [ N, B, H ]
        cp, batch, q_heads, v_head_dim
    Return:
        output: [ B, H, D ]
        lse   : [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)

    # shape = [N]
    lse_offsets = num_n_offsets * lses_stride_N + batch_idx * \
        lses_stride_B + head_idx * lses_stride_H

    # calc final lse
    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float('inf')), -float('inf'), lse)
    lse_max = tl.max(lse, axis=0)
    lse -= lse_max
    lse_exp = tl.exp(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    lse = tl.log(lse_acc)
    lse += lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    # shape = [D]
    output_offsets = batch_idx * outputs_stride_B + \
                    head_idx * outputs_stride_H + \
                    d_offsets * outputs_stride_D

    # correct output
    lse_offset = lse_idx * lses_stride_N + batch_idx * \
        lses_stride_B + head_idx * lses_stride_H
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float('inf')),
        -float('inf'), lse_finally)
    factor = tl.exp(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor

    tl.store(new_output_ptr + output_offsets, output)


class CPTritonContext:
    """ The CPTritonContext is used to avoid recompilation of the Triton JIT.
    """

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(out: torch.Tensor, lses: torch.Tensor, cp_rank: int,
                     ctx: CPTritonContext):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        output: [ B, H, D ]
        lses   : [ N, B, H ]
    Return:
        output: [ B, H, D ]
        lse   : [ B, H ]
    """
    if ctx is None:
        ctx = CPTritonContext()

    lse = torch.empty_like(lses[0])

    grid = (out.shape[0], out.shape[1], 1)
    regular_args = (out, out, lses, lse, *out.stride(), *lses.stride(),
                    cp_rank)
    const_args = {
        "HEAD_DIM": out.shape[-1],
        "N_ROUNDED": lses.shape[0],
    }

    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args,
                    **const_args)
    return out, lse


def cp_lse_ag_out_rs(cp_attn_out: torch.Tensor,
                     cp_attn_lse: torch.Tensor,
                     cp_group: GroupCoordinator,
                     ctx: CPTritonContext = None):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    if ctx is None:
        ctx = CPTritonContext()

    lses = torch.empty((cp_group.world_size, ) + cp_attn_lse.shape,
                       dtype=cp_attn_lse.dtype,
                       device=cp_attn_lse.device)

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = cp_group.all_gather(cp_attn_lse, dim=0).view_as(lses)
    out, _ = correct_attn_out(cp_attn_out, lses, cp_group.rank_in_group, ctx)
    assert out.is_contiguous()
    out = cp_group.reduce_scatter(out, dim=1)
    return out
