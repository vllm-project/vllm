# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused kernel for DCP split-Q expansion.

With DCP + speculative decoding, multi-token decode queries need per-token
DCP-local seq_lens and expanded block tables.  This kernel fuses the
global -> local seq_len conversion and block_table row repetition into a
single pass.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _dcp_split_q_kernel(
    # Inputs
    global_seq_lens_ptr,  # [num_decodes]
    block_table_ptr,  # [num_decodes, ncols]
    bt_stride,  # block_table stride for dim 0
    # Outputs
    out_seq_lens_ptr,  # [num_decode_tokens]
    out_block_table_ptr,  # [num_decode_tokens, ncols]
    out_bt_stride,  # out_block_table stride for dim 0
    # Scalars
    ncols,
    tokens_per_req: tl.constexpr,
    NCOLS_PAD: tl.constexpr,
    dcp_world_size: tl.constexpr,
    dcp_rank: tl.constexpr,
    interleave: tl.constexpr,
):
    out_idx = tl.program_id(0)
    req_idx = out_idx // tokens_per_req
    t = out_idx % tokens_per_req

    # --- seq_len: apply causal offset, convert global -> DCP-local ---
    global_sl = tl.load(global_seq_lens_ptr + req_idx)
    g = global_sl + t - (tokens_per_req - 1)

    virtual = dcp_world_size * interleave
    base = g // virtual * interleave
    rem = g - base * dcp_world_size - dcp_rank * interleave
    rem = tl.maximum(rem, 0)
    rem = tl.minimum(rem, interleave)
    tl.store(out_seq_lens_ptr + out_idx, base + rem)

    # --- block_table: copy row from req_idx to out_idx ---
    col_offsets = tl.arange(0, NCOLS_PAD)
    mask = col_offsets < ncols
    row = tl.load(
        block_table_ptr + req_idx * bt_stride + col_offsets,
        mask=mask,
    )
    tl.store(
        out_block_table_ptr + out_idx * out_bt_stride + col_offsets,
        row,
        mask=mask,
    )


def dcp_split_q(
    global_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_decodes: int,
    tokens_per_req: int,
    dcp_world_size: int,
    dcp_rank: int,
    interleave: int,
    out_seq_lens: torch.Tensor | None = None,
    out_block_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token DCP-local seq_lens and expand block_table.

    Args:
        global_seq_lens: Global (pre-DCP) seq_lens, shape [num_decodes].
        block_table: Block table, shape [num_decodes, ncols].
        num_decodes: Number of decode requests.
        tokens_per_req: Tokens per request (1 + num_speculative_tokens).
        dcp_world_size: DCP group size.
        dcp_rank: This rank's position in the DCP group.
        interleave: cp_kv_cache_interleave_size.
        out_seq_lens: Optional pre-allocated output [num_decode_tokens].
        out_block_table: Optional pre-allocated output [num_decode_tokens, ncols].

    Returns:
        (out_seq_lens, out_block_table) with correct per-token values.
    """
    num_decode_tokens = num_decodes * tokens_per_req
    ncols = block_table.shape[1]

    if out_seq_lens is None:
        out_seq_lens = torch.empty(
            num_decode_tokens,
            device=global_seq_lens.device,
            dtype=global_seq_lens.dtype,
        )
    if out_block_table is None:
        out_block_table = torch.empty(
            num_decode_tokens,
            ncols,
            device=block_table.device,
            dtype=block_table.dtype,
        )

    ncols_padded = triton.next_power_of_2(ncols)

    _dcp_split_q_kernel[(num_decode_tokens,)](
        global_seq_lens,
        block_table,
        block_table.stride(0),
        out_seq_lens,
        out_block_table,
        out_block_table.stride(0),
        ncols,
        tokens_per_req=tokens_per_req,
        NCOLS_PAD=ncols_padded,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        interleave=interleave,
    )

    return out_seq_lens, out_block_table
